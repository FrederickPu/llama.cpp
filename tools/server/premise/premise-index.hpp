#pragma once

#include "json.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// FAISS stores vectors in IndexFlatIP and exposes stable declaration IDs through
// IndexIDMap2. modules[m].declaration_ids and rows use those stable IDs, never internal
// IndexFlat offsets. Cache: --index-vecs (FAISS) + --index-names (PMNAME02).
struct PremiseIndex {
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
        std::vector<faiss::idx_t> declaration_ids;
    };

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
        if (dim > 0 && !faiss) {
            faiss = make_faiss(dim);
        }

        auto existing = modules.find(module);
        if (existing != modules.end() && !existing->second.declaration_ids.empty()) {
            remove_ids(existing->second.declaration_ids);
        }

        ModuleEntry & entry = modules[module];
        entry.version_token = version_token;
        entry.imports = imports;
        entry.declaration_ids.clear();

        std::unordered_set<std::string> seen;
        std::vector<float> packed;
        std::vector<faiss::idx_t> ids;
        packed.reserve(replacements.size() * (size_t) std::max(dim, 0));
        ids.reserve(replacements.size());

        for (const auto & c : replacements) {
            if (c.name.empty() || !seen.insert(c.name).second) {
                continue;
            }
            if ((int) c.embedding.size() != dim) {
                throw std::runtime_error("embedding dimension mismatch");
            }
            if (next_id == std::numeric_limits<faiss::idx_t>::max()) {
                throw std::runtime_error("stable declaration ID space exhausted");
            }
            const faiss::idx_t id = next_id++;
            ids.push_back(id);
            packed.insert(packed.end(), c.embedding.begin(), c.embedding.end());
            entry.declaration_ids.push_back(id);
            rows.emplace(id, Row{c.name, module});
        }

        if (!ids.empty()) {
            faiss->add_with_ids((faiss::idx_t) ids.size(), packed.data(), ids.data());
        }
        if (!vecs_path.empty() && !names_path.empty()) {
            dirty = true;
        }
    }

    std::vector<Hit> search_global(const float * query, int top_k) const {
        std::lock_guard<std::mutex> lock(mu);
        return search_faiss(query, top_k, {});
    }

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

        // Empty scoped closure means no indexed candidates, not global search.
        std::vector<Hit> hits;
        if (!candidate_rows.empty()) {
            hits = search_faiss(query, top_k, candidate_rows);
        }

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

    // Persist after every successful /cache. The FAISS ID map is written
    // directly; no second in-memory vector index is built.
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
    inline static constexpr char NAMES_MAGIC[8] = {'P', 'M', 'N', 'A', 'M', 'E', '0', '4'};

    struct Row {
        std::string name;
        std::string module;
    };

    int dim = 0;
    faiss::idx_t next_id = 0;
    std::unordered_map<faiss::idx_t, Row> rows;
    std::unordered_map<std::string, ModuleEntry> modules;
    std::unique_ptr<faiss::IndexIDMap2> faiss;
    std::string vecs_path;
    std::string names_path;
    bool dirty = false;
    mutable std::mutex mu;

    void clear(int model_dim) {
        dim = model_dim;
        next_id = 0;
        rows.clear();
        modules.clear();
        faiss = make_faiss(model_dim);
        dirty = false;
    }

    void remove_ids(const std::vector<faiss::idx_t> & ids) {
        faiss::IDSelectorBatch remove(ids.size(), ids.data());
        if (faiss && faiss->remove_ids(remove) != ids.size()) {
            throw std::runtime_error("failed to remove module IDs from FAISS");
        }
        for (faiss::idx_t id : ids) {
            rows.erase(id);
        }
    }

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
            // IndexIDMap translates this external-ID selector for IndexFlat.
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
            auto row = rows.find(labels[(size_t) i]);
            if (row == rows.end()) {
                continue;
            }
            out.push_back({row->second.name, row->second.module, scores[(size_t) i]});
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
        for (faiss::idx_t id : it->second.declaration_ids) {
            auto row = rows.find(id);
            if (row != rows.end() && !row->second.name.empty() &&
                    taken_names.insert(row->second.name).second) {
                candidate_rows.push_back(id);
            }
        }
    }

    void load_pair(std::istream & names_in, const std::string & vecs_file, int model_dim) {
        char magic[8] = {};
        names_in.read(magic, 8);
        if (!names_in || std::memcmp(magic, NAMES_MAGIC, 8) != 0) {
            throw std::runtime_error("bad names magic");
        }

        read_i64(names_in, next_id);
        if (next_id < 0) {
            throw std::runtime_error("invalid next declaration ID");
        }

        uint32_t n_modules = 0;
        read_u32(names_in, n_modules);
        std::unordered_set<std::string> seen_names;
        std::unordered_set<faiss::idx_t> seen_ids;

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
                faiss::idx_t id = 0;
                read_i64(names_in, id);
                std::string name;
                read_str(names_in, name);
                if (id < 0 || id >= next_id || name.empty() || !seen_ids.insert(id).second ||
                        !seen_names.insert(mod + "\n" + name).second) {
                    throw std::runtime_error("invalid declaration identity");
                }
                entry.declaration_ids.push_back(id);
                rows.emplace(id, Row{std::move(name), mod});
            }
            modules[std::move(mod)] = std::move(entry);
        }

        faiss = read_id_map(vecs_file.c_str());
        if ((int) faiss->d != model_dim || (size_t) faiss->ntotal != rows.size()) {
            throw std::runtime_error("vecs/names size mismatch");
        }
        if (faiss->id_map.size() != rows.size()) {
            throw std::runtime_error("FAISS ID map size mismatch");
        }
        for (faiss::idx_t id : faiss->id_map) {
            if (rows.find(id) == rows.end()) {
                throw std::runtime_error("FAISS contains unknown declaration ID");
            }
        }
        faiss->construct_rev_map();
        dim = model_dim;
    }

    void save_pair(const std::string & vecs_tmp, const std::string & names_tmp) const {
        std::ofstream names(names_tmp, std::ios::binary | std::ios::trunc);
        if (!names || !faiss) {
            throw std::runtime_error("cannot open cache temps");
        }

        names.write(NAMES_MAGIC, 8);
        write_i64(names, next_id);
        write_u32(names, (uint32_t) modules.size());
        for (const auto & item : modules) {
            write_str(names, item.first);
            write_str(names, item.second.version_token);
            write_u32(names, (uint32_t) item.second.imports.size());
            for (const auto & imp : item.second.imports) {
                write_str(names, imp);
            }
            write_u32(names, (uint32_t) item.second.declaration_ids.size());
            for (faiss::idx_t id : item.second.declaration_ids) {
                auto row = rows.find(id);
                if (row == rows.end()) {
                    throw std::runtime_error("module references unknown declaration ID");
                }
                write_i64(names, id);
                write_str(names, row->second.name);
            }
        }
        names.close();
        if (!names) {
            throw std::runtime_error("cannot finish names file");
        }
        faiss::write_index(faiss.get(), vecs_tmp.c_str());
    }

    static std::unique_ptr<faiss::IndexIDMap2> make_faiss(int d) {
        if (d <= 0) {
            return nullptr;
        }
        auto out = std::make_unique<faiss::IndexIDMap2>(new faiss::IndexFlatIP(d));
        out->own_fields = true;
        return out;
    }

    static std::unique_ptr<faiss::IndexIDMap2> read_id_map(const char * path) {
        std::unique_ptr<faiss::Index> idx(faiss::read_index(path));
        auto * map = dynamic_cast<faiss::IndexIDMap2 *>(idx.get());
        if (!map || map->metric_type != faiss::METRIC_INNER_PRODUCT) {
            throw std::runtime_error(std::string("expected IndexIDMap2(IndexFlatIP): ") + path);
        }
        if (!dynamic_cast<faiss::IndexFlat *>(map->index)) {
            throw std::runtime_error(std::string("expected flat ID map storage: ") + path);
        }
        idx.release();
        return std::unique_ptr<faiss::IndexIDMap2>(map);
    }

    static void read_u32(std::istream & in, uint32_t & value) {
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!in) {
            throw std::runtime_error("truncated u32");
        }
    }

    static void write_u32(std::ostream & out, uint32_t value) {
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    static void read_i64(std::istream & in, faiss::idx_t & value) {
        static_assert(sizeof(faiss::idx_t) == sizeof(int64_t));
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!in) {
            throw std::runtime_error("truncated i64");
        }
    }

    static void write_i64(std::ostream & out, faiss::idx_t value) {
        static_assert(sizeof(faiss::idx_t) == sizeof(int64_t));
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    static void read_str(std::istream & in, std::string & value) {
        uint32_t n = 0;
        read_u32(in, n);
        value.resize(n);
        if (n) {
            in.read(value.data(), n);
        }
        if (!in) {
            throw std::runtime_error("truncated string");
        }
    }

    static void write_str(std::ostream & out, const std::string & value) {
        write_u32(out, (uint32_t) value.size());
        out.write(value.data(), (std::streamsize) value.size());
    }
};
