#pragma once

#include "json.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// rows[i] <-> FAISS label i. Embeddings live only in FAISS.
// modules[m].row_ids = FAISS labels owned by module m.
// Cache: --index-vecs (FAISS) + --index-names (PMNAME01); name order == FAISS order.
struct PremiseIndex {
    // Inbound only (replace_module / local /select). Not stored long-term.
    struct Candidate {
        std::string name;
        std::string module;
        std::vector<float> embedding;
    };

    struct Hit {
        std::string name;
        std::string module;
        float score = 0.0f;
    };

    struct ModuleEntry {
        std::string version_token;
        std::vector<std::string> imports;
        std::vector<faiss::idx_t> row_ids;
    };

    // Kept as IndexedDeclaration for existing call sites.
    using IndexedDeclaration = Candidate;

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu);
        return rows.size();
    }

    int embedding_dim() const {
        std::lock_guard<std::mutex> lock(mu);
        return dim;
    }

    void initialize_empty(int model_dim) {
        std::lock_guard<std::mutex> lock(mu);
        vecs_path.clear();
        names_path.clear();
        clear(model_dim);
    }

    void load_cache(const std::string & vecs_file, const std::string & names_file, int model_dim) {
        std::lock_guard<std::mutex> lock(mu);
        vecs_path = vecs_file;
        names_path = names_file;
        clear(model_dim);

        std::ifstream in(names_file, std::ios::binary);
        if (!in) {
            fprintf(stderr, "premise cache: starting empty\n");
            return;
        }
        try {
            load_pair(in, vecs_file, model_dim);
            dirty = false;
            fprintf(stderr, "premise cache: loaded %zu rows\n", rows.size());
        } catch (const std::exception & e) {
            clear(model_dim);
            fprintf(stderr, "premise cache: ignoring corrupt cache (%s)\n", e.what());
        }
    }

    std::string get_module_version(const std::string & module) const {
        std::lock_guard<std::mutex> lock(mu);
        auto it = modules.find(module);
        return it == modules.end() ? std::string() : it->second.version_token;
    }

    void replace_module(const std::string & module,
                        const std::string & version_token,
                        const std::vector<std::string> & imports,
                        const std::vector<Candidate> & replacements) {
        std::lock_guard<std::mutex> lock(mu);
        if (dim <= 0 && !replacements.empty()) {
            dim = (int) replacements.front().embedding.size();
        }

        // Build next FAISS + row table: keep other modules, then this module's replacements.
        auto next_faiss = make_faiss(dim);
        std::vector<Row> next_rows;
        std::unordered_map<std::string, ModuleEntry> next_modules;

        for (const auto & item : modules) {
            if (item.first != module) {
                next_modules[item.first] = {item.second.version_token, item.second.imports, {}};
            }
        }
        next_modules[module] = {version_token, imports, {}};

        std::vector<float> emb((size_t) std::max(dim, 0));
        for (faiss::idx_t r = 0; r < (faiss::idx_t) rows.size(); ++r) {
            if (rows[(size_t) r].module == module) {
                continue;
            }
            if (dim > 0 && faiss) {
                emb.resize((size_t) dim);
                faiss->reconstruct(r, emb.data());
                next_faiss->add(1, emb.data());
            }
            next_modules[rows[(size_t) r].module].row_ids.push_back((faiss::idx_t) next_rows.size());
            next_rows.push_back(rows[(size_t) r]);
        }

        std::unordered_set<std::string> seen;
        for (const auto & c : replacements) {
            if (c.name.empty() || !seen.insert(c.name).second) {
                continue;
            }
            if ((int) c.embedding.size() != dim) {
                throw std::runtime_error("embedding dimension mismatch");
            }
            next_faiss->add(1, c.embedding.data());
            next_modules[module].row_ids.push_back((faiss::idx_t) next_rows.size());
            next_rows.push_back({c.name, module});
        }

        rows = std::move(next_rows);
        modules = std::move(next_modules);
        faiss = std::move(next_faiss);
        if (!vecs_path.empty() && !names_path.empty()) {
            dirty = true;
        }
    }

    std::vector<Hit> search_global(const float * query, int top_k) const {
        std::lock_guard<std::mutex> lock(mu);
        return search_faiss(query, top_k, {});
    }

    // Large set: FAISS top-k among import-closure candidate_rows.
    // Small set: score request-local candidates in place (never inserted into FAISS).
    // Then merge and keep overall top-k.
    std::vector<Hit> search_scoped(const float * query,
                                   const std::vector<std::string> & import_roots,
                                   const std::vector<Candidate> & locals,
                                   int top_k) const {
        std::lock_guard<std::mutex> lock(mu);
        if (top_k <= 0) {
            return {};
        }

        std::vector<faiss::idx_t> candidate_rows;
        std::unordered_set<std::string> taken_names;
        std::unordered_set<std::string> visited;
        for (const auto & root : import_roots) {
            collect_candidate_rows(root, visited, taken_names, candidate_rows);
        }

        std::vector<Hit> hits = search_faiss(query, top_k, candidate_rows);

        for (const auto & local : locals) {
            if (local.name.empty() || !taken_names.insert(local.name).second) {
                continue;
            }
            if ((int) local.embedding.size() != dim) {
                throw std::runtime_error("local embedding dimension mismatch");
            }
            float score = 0.0f;
            for (int i = 0; i < dim; ++i) {
                score += query[i] * local.embedding[(size_t) i];
            }
            hits.push_back({local.name, local.module, score});
        }

        if (hits.empty()) {
            return {};
        }
        const int n = std::min(top_k, (int) hits.size());
        std::partial_sort(hits.begin(), hits.begin() + n, hits.end(),
                [](const Hit & a, const Hit & b) { return a.score > b.score; });
        hits.resize((size_t) n);
        return hits;
    }

    // Write vecs/names when dirty. Called after every successful /cache.
    void flush() {
        std::lock_guard<std::mutex> lock(mu);
        if (!dirty || vecs_path.empty() || names_path.empty()) {
            return;
        }

        const std::string vecs_tmp = vecs_path + ".tmp";
        const std::string names_tmp = names_path + ".tmp";
        try {
            save_pair(vecs_tmp, names_tmp);
            std::remove(vecs_path.c_str());
            std::remove(names_path.c_str());
            if (std::rename(vecs_tmp.c_str(), vecs_path.c_str()) != 0 ||
                    std::rename(names_tmp.c_str(), names_path.c_str()) != 0) {
                throw std::runtime_error("failed to install cache files");
            }
        } catch (const std::exception & e) {
            std::remove(vecs_tmp.c_str());
            std::remove(names_tmp.c_str());
            fprintf(stderr, "premise cache: save failed (%s)\n", e.what());
            return;
        }
        dirty = false;
        fprintf(stderr, "premise cache: saved %zu rows\n", rows.size());
    }

private:
    inline static constexpr char NAMES_MAGIC[8] = {'P', 'M', 'N', 'A', 'M', 'E', '0', '1'};

    struct Row {
        std::string name;
        std::string module;
    };

    int dim = 0;
    std::vector<Row> rows;
    std::unordered_map<std::string, ModuleEntry> modules;
    std::unique_ptr<faiss::IndexFlat> faiss;
    std::string vecs_path;
    std::string names_path;
    bool dirty = false;
    mutable std::mutex mu;

    void clear(int model_dim) {
        dim = model_dim;
        rows.clear();
        modules.clear();
        faiss = make_faiss(model_dim);
        dirty = false;
    }

    // candidate_rows empty => full index; else FAISS top-k restricted to those labels.
    std::vector<Hit> search_faiss(const float * query, int top_k,
                                  const std::vector<faiss::idx_t> & candidate_rows) const {
        if (!faiss || top_k <= 0 || rows.empty()) {
            return {};
        }

        const bool restricted = !candidate_rows.empty();
        const int n = restricted
            ? std::min(top_k, (int) candidate_rows.size())
            : std::min(top_k, (int) rows.size());
        if (n <= 0) {
            return {};
        }

        std::vector<float> scores((size_t) n);
        std::vector<faiss::idx_t> labels((size_t) n);
        if (restricted) {
            faiss::IDSelectorBatch allow(candidate_rows.size(), candidate_rows.data());
            faiss::SearchParameters params;
            params.sel = &allow;
            faiss->search(1, query, n, scores.data(), labels.data(), &params);
        } else {
            faiss->search(1, query, n, scores.data(), labels.data());
        }

        std::vector<Hit> out;
        out.reserve((size_t) n);
        for (int i = 0; i < n; ++i) {
            const faiss::idx_t r = labels[(size_t) i];
            if (r < 0 || r >= (faiss::idx_t) rows.size()) {
                continue; // FAISS pads unused slots with -1
            }
            out.push_back({rows[(size_t) r].name, rows[(size_t) r].module, scores[(size_t) i]});
        }
        return out;
    }

    void collect_candidate_rows(const std::string & module,
                                std::unordered_set<std::string> & visited,
                                std::unordered_set<std::string> & taken_names,
                                std::vector<faiss::idx_t> & candidate_rows) const {
        if (!visited.insert(module).second) {
            return;
        }
        auto it = modules.find(module);
        if (it == modules.end()) {
            return;
        }
        for (const auto & imp : it->second.imports) {
            collect_candidate_rows(imp, visited, taken_names, candidate_rows);
        }
        for (faiss::idx_t r : it->second.row_ids) {
            const auto & name = rows[(size_t) r].name;
            if (!name.empty() && taken_names.insert(name).second) {
                candidate_rows.push_back(r);
            }
        }
    }

    // names file: PMNAME01 | modules{ name, version, imports[], decl_names[] }
    // flattened decl_names order == FAISS row order
    void load_pair(std::istream & names_in, const std::string & vecs_file, int model_dim) {
        char magic[8] = {};
        names_in.read(magic, 8);
        if (!names_in || std::memcmp(magic, NAMES_MAGIC, 8) != 0) {
            throw std::runtime_error("bad names magic");
        }

        uint32_t n_modules = 0;
        read_u32(names_in, n_modules);
        std::unordered_set<std::string> seen;
        for (uint32_t i = 0; i < n_modules; ++i) {
            std::string mod, ver;
            read_str(names_in, mod);
            read_str(names_in, ver);
            ModuleEntry entry{std::move(ver), {}, {}};

            uint32_t n_imports = 0;
            read_u32(names_in, n_imports);
            entry.imports.resize(n_imports);
            for (uint32_t j = 0; j < n_imports; ++j) {
                read_str(names_in, entry.imports[j]);
            }

            uint32_t n_decls = 0;
            read_u32(names_in, n_decls);
            for (uint32_t j = 0; j < n_decls; ++j) {
                std::string name;
                read_str(names_in, name);
                if (name.empty() || !seen.insert(mod + "\n" + name).second) {
                    throw std::runtime_error("invalid declaration identity");
                }
                entry.row_ids.push_back((faiss::idx_t) rows.size());
                rows.push_back({std::move(name), mod});
            }
            modules[std::move(mod)] = std::move(entry);
        }

        auto index = read_faiss(vecs_file.c_str());
        if ((int) index->d != model_dim || (size_t) index->ntotal != rows.size()) {
            throw std::runtime_error("vecs/names size mismatch");
        }
        dim = model_dim;
        faiss = std::move(index);
    }

    void save_pair(const std::string & vecs_tmp, const std::string & names_tmp) const {
        auto out = make_faiss(dim);
        std::ofstream names(names_tmp, std::ios::binary | std::ios::trunc);
        if (!names || !out) {
            throw std::runtime_error("cannot open cache temps");
        }

        names.write(NAMES_MAGIC, 8);
        write_u32(names, (uint32_t) modules.size());
        std::vector<float> emb((size_t) dim);
        for (const auto & item : modules) {
            write_str(names, item.first);
            write_str(names, item.second.version_token);
            write_u32(names, (uint32_t) item.second.imports.size());
            for (const auto & imp : item.second.imports) {
                write_str(names, imp);
            }
            write_u32(names, (uint32_t) item.second.row_ids.size());
            for (faiss::idx_t r : item.second.row_ids) {
                write_str(names, rows[(size_t) r].name);
                faiss->reconstruct(r, emb.data());
                out->add(1, emb.data());
            }
        }
        names.close();
        if (!names) {
            throw std::runtime_error("cannot finish names file");
        }
        faiss::write_index(out.get(), vecs_tmp.c_str());
    }

    static std::unique_ptr<faiss::IndexFlat> make_faiss(int d) {
        return d > 0 ? std::make_unique<faiss::IndexFlatIP>(d) : nullptr;
    }

    static std::unique_ptr<faiss::IndexFlat> read_faiss(const char * path) {
        std::unique_ptr<faiss::Index> idx(faiss::read_index(path));
        auto * flat = dynamic_cast<faiss::IndexFlat *>(idx.get());
        if (!flat || flat->metric_type != faiss::METRIC_INNER_PRODUCT) {
            throw std::runtime_error(std::string("expected IndexFlatIP: ") + path);
        }
        idx.release();
        return std::unique_ptr<faiss::IndexFlat>(flat);
    }

    static void read_u32(std::istream & in, uint32_t & v) {
        in.read(reinterpret_cast<char *>(&v), 4);
        if (!in) {
            throw std::runtime_error("truncated u32");
        }
    }
    static void write_u32(std::ostream & out, uint32_t v) {
        out.write(reinterpret_cast<const char *>(&v), 4);
    }
    static void read_str(std::istream & in, std::string & s) {
        uint32_t n = 0;
        read_u32(in, n);
        s.resize(n);
        if (n) {
            in.read(s.data(), n);
        }
        if (!in) {
            throw std::runtime_error("truncated string");
        }
    }
    static void write_str(std::ostream & out, const std::string & s) {
        write_u32(out, (uint32_t) s.size());
        out.write(s.data(), (std::streamsize) s.size());
    }
};
