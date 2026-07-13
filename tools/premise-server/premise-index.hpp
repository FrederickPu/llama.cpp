#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "json.hpp"

#ifdef LLAMA_PREMISE_USE_FAISS
#include <faiss/IndexFlat.h>
#endif

struct PremiseIndex {
    int n_premises = 0;
    int dim        = 0;
    std::vector<float>       vectors;  // [n_premises * dim], row-major, L2-normalised
    std::vector<std::string> strings;
#ifdef LLAMA_PREMISE_USE_FAISS
    std::unique_ptr<faiss::IndexFlatIP> faiss_index;
#endif

    void load(const char * vec_path, const char * str_path, int max_premises = 0) {
        // Load premise strings
        {
            std::ifstream f(str_path);
            if (!f) throw std::runtime_error(std::string("cannot open ") + str_path);
            nlohmann::json j; f >> j;
            strings = j.get<std::vector<std::string>>();
        }

        // Load premise vectors
        {
            std::ifstream f(vec_path, std::ios::binary);
            if (!f) throw std::runtime_error(std::string("cannot open ") + vec_path);
            // Header: n_premises (int32), dim (int32)
            int32_t n, d;
            f.read(reinterpret_cast<char *>(&n), sizeof(n));
            f.read(reinterpret_cast<char *>(&d), sizeof(d));
            n_premises = (max_premises > 0 && max_premises < n) ? max_premises : n;
            dim        = d;
            vectors.resize((size_t)n_premises * dim);
            f.read(reinterpret_cast<char *>(vectors.data()),
                   (size_t)n_premises * dim * sizeof(float));
        }

        if ((int)strings.size() > n_premises) strings.resize(n_premises);
        if ((int)strings.size() < n_premises) {
            throw std::runtime_error("premise metadata has fewer entries than premise_vectors.bin");
        }
#ifdef LLAMA_PREMISE_USE_FAISS
        faiss_index = std::make_unique<faiss::IndexFlatIP>(dim);
        if (n_premises > 0) {
            faiss_index->add(n_premises, vectors.data());
        }
        fprintf(stderr, "premise index: %d premises, dim=%d, backend=faiss\n", n_premises, dim);
#else
        fprintf(stderr, "premise index: %d premises, dim=%d, backend=bruteforce\n", n_premises, dim);
#endif
    }

    // Brute-force cosine similarity (inner product on L2-normalised vectors).
    // Returns top_k (premise_string, score) pairs, sorted by descending score.
    std::vector<std::pair<std::string, float>>
    search(const float * query, int top_k) const {
        int k = std::min(top_k, n_premises);
        if (k <= 0) {
            return {};
        }
#ifdef LLAMA_PREMISE_USE_FAISS
        if (faiss_index) {
            std::vector<float> scores(k);
            std::vector<faiss::idx_t> labels(k);
            faiss_index->search(1, query, k, scores.data(), labels.data());

            std::vector<std::pair<std::string, float>> out;
            out.reserve(k);
            for (int i = 0; i < k; ++i) {
                if (labels[i] >= 0 && labels[i] < (faiss::idx_t) strings.size()) {
                    out.push_back({strings[(size_t) labels[i]], scores[i]});
                }
            }
            return out;
        }
#endif
        std::vector<std::pair<float, int>> scores(n_premises);
        for (int i = 0; i < n_premises; ++i) {
            float dot = 0.0f;
            const float * row = vectors.data() + (size_t)i * dim;
            for (int d = 0; d < dim; ++d) dot += query[d] * row[d];
            scores[i] = {dot, i};
        }
        std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                          [](const auto & a, const auto & b){ return a.first > b.first; });
        std::vector<std::pair<std::string, float>> out;
        out.reserve(k);
        for (int i = 0; i < k; ++i)
            out.push_back({strings[scores[i].second], scores[i].first});
        return out;
    }
};

// Persistent embedding cache used by standalone premise-server
// (PremiseMode::Embedding). Lean owns elaboration and module freshness: it sends
// /version requests, and if a module token is stale it sends /cache with module,
// imports, fully-qualified declaration names, and declaration strings. The server
// embeds those strings but does not persist them.
//
// On disk this is two aligned tables:
//   - premise_vectors.bin: int32 n_rows, int32 dim, then n_rows * dim floats
//   - premise_names.bin: binary metadata grouped by defining module
//
// premise_names.bin format (little-endian):
//   magic "PMNAME01"
//   u32 n_modules
//   repeated module blocks:
//     str module_name
//     str version_token
//     u32 n_imports, repeated str import_name
//     u32 n_decls, repeated str declaration_name
//
// String encoding is u32 byte length followed by UTF-8 bytes. Declaration names
// are stored under the module that defines them. The declaration traversal order
// in premise_names.bin is exactly the vector row order in premise_vectors.bin.
struct PremiseEmbedCache {
    static constexpr size_t SAVE_EVERY   = 1024; // unsaved rows that force a save
    static constexpr int    SAVE_SECONDS = 60;   // dirty-cache flush interval
    inline static constexpr char META_MAGIC[8] = {'P', 'M', 'N', 'A', 'M', 'E', '0', '1'};

    struct Entry {
        std::string module;
        std::string name;
        std::vector<float> embedding;
    };

    struct ModuleMeta {
        std::string version_token;
        std::vector<std::string> imports;
        std::vector<size_t> rows;
    };

    struct ModuleSnapshot {
        std::string module;
        std::string version_token;
        std::vector<std::string> imports;
        std::vector<std::pair<std::string, std::vector<float>>> declarations;
    };

    std::string vec_path;
    std::string meta_path;
    int dim = 0;

    std::mutex mu;
    std::vector<Entry>                       entries; // insertion order == vector row order
    std::unordered_map<std::string, size_t>  index;   // module + name -> row
    std::unordered_map<std::string, ModuleMeta> modules;
    size_t n_dirty = 0;
    std::chrono::steady_clock::time_point last_save = std::chrono::steady_clock::now();

    static std::string key(const std::string & module, const std::string & name) {
        return module + "\n" + name;
    }

    static bool read_u32(std::istream & in, uint32_t & out) {
        in.read(reinterpret_cast<char *>(&out), sizeof(out));
        return !!in;
    }

    static bool read_string(std::istream & in, std::string & out) {
        uint32_t n = 0;
        if (!read_u32(in, n)) {
            return false;
        }
        out.resize(n);
        if (n > 0) {
            in.read(out.data(), n);
        }
        return !!in;
    }

    static void write_u32(std::ostream & out, uint32_t value) {
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    static void write_string(std::ostream & out, const std::string & value) {
        write_u32(out, (uint32_t) value.size());
        out.write(value.data(), (std::streamsize) value.size());
    }

    void load(const std::string & vec, const std::string & meta, int model_dim) {
        vec_path = vec;
        meta_path = meta;
        dim      = model_dim;
        std::ifstream fs(meta_path, std::ios::binary);
        std::ifstream fv(vec_path, std::ios::binary);
        if (!fs || !fv) {
            fprintf(stderr, "premise cache: starting empty (index files not present yet)\n");
            return;
        }
        std::vector<std::pair<std::string, std::string>> loaded;
        char magic[sizeof(META_MAGIC)] = {};
        fs.read(magic, sizeof(magic));
        if (!fs || std::memcmp(magic, META_MAGIC, sizeof(META_MAGIC)) != 0) {
            fprintf(stderr, "premise cache: ignoring unsupported metadata file %s\n", meta_path.c_str());
            return;
        }

        uint32_t n_modules = 0;
        if (!read_u32(fs, n_modules)) {
            fprintf(stderr, "premise cache: ignoring truncated metadata file %s\n", meta_path.c_str());
            return;
        }
        for (uint32_t i = 0; i < n_modules; ++i) {
            std::string module;
            ModuleMeta meta;
            if (!read_string(fs, module) || !read_string(fs, meta.version_token)) {
                fprintf(stderr, "premise cache: ignoring truncated metadata file %s\n", meta_path.c_str());
                return;
            }
            uint32_t n_imports = 0;
            if (!read_u32(fs, n_imports)) {
                fprintf(stderr, "premise cache: ignoring truncated metadata file %s\n", meta_path.c_str());
                return;
            }
            meta.imports.reserve(n_imports);
            for (uint32_t j = 0; j < n_imports; ++j) {
                std::string imported;
                if (!read_string(fs, imported)) {
                    fprintf(stderr, "premise cache: ignoring truncated metadata file %s\n", meta_path.c_str());
                    return;
                }
                meta.imports.push_back(std::move(imported));
            }
            uint32_t n_decls = 0;
            if (!read_u32(fs, n_decls)) {
                fprintf(stderr, "premise cache: ignoring truncated metadata file %s\n", meta_path.c_str());
                return;
            }
            modules.emplace(module, std::move(meta));
            for (uint32_t j = 0; j < n_decls; ++j) {
                std::string name;
                if (!read_string(fs, name)) {
                    fprintf(stderr, "premise cache: ignoring truncated metadata file %s\n", meta_path.c_str());
                    return;
                }
                loaded.push_back({module, std::move(name)});
            }
        }
        int32_t n = 0, d = 0;
        fv.read(reinterpret_cast<char *>(&n), sizeof(n));
        fv.read(reinterpret_cast<char *>(&d), sizeof(d));
        if (!fv || d != model_dim) {
            fprintf(stderr, "premise cache: ignoring %s (dim %d != model dim %d)\n",
                    vec_path.c_str(), d, model_dim);
            return;
        }
        size_t count = std::min(loaded.size(), (size_t) std::max<int32_t>(0, n));
        for (size_t i = 0; i < count; ++i) {
            std::vector<float> row(dim);
            fv.read(reinterpret_cast<char *>(row.data()), (size_t) dim * sizeof(float));
            if (!fv) { count = i; break; }
            const auto & module = loaded[i].first;
            const auto & name = loaded[i].second;
            if (index.emplace(key(module, name), entries.size()).second) {
                modules[module].rows.push_back(entries.size());
                entries.push_back({module, name, std::move(row)});
            }
        }
        if (entries.size() != loaded.size() || (int32_t) count != n) {
            n_dirty = entries.size();
        }
        fprintf(stderr, "premise cache: loaded %zu embeddings from %s\n", entries.size(), vec_path.c_str());
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return entries.size();
    }

    std::vector<ModuleSnapshot> snapshot_modules() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<ModuleSnapshot> out;
        out.reserve(modules.size());
        for (const auto & item : modules) {
            ModuleSnapshot snap;
            snap.module = item.first;
            snap.version_token = item.second.version_token;
            snap.imports = item.second.imports;
            snap.declarations.reserve(item.second.rows.size());
            for (size_t row : item.second.rows) {
                if (row < entries.size()) {
                    snap.declarations.push_back({entries[row].name, entries[row].embedding});
                }
            }
            out.push_back(std::move(snap));
        }
        return out;
    }

    bool find(const std::string & module, const std::string & name, std::vector<float> & out) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = index.find(key(module, name));
        if (it == index.end()) {
            return false;
        }
        out = entries[it->second].embedding;
        return true;
    }

    void replace_module(const std::string & module,
                        const std::string & token,
                        const std::vector<std::string> & imports,
                        const std::vector<std::string> & names,
                        const std::vector<std::vector<float>> & embeddings) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<Entry> next;
        next.reserve(entries.size() + names.size());
        std::unordered_map<std::string, size_t> next_index;
        std::unordered_map<std::string, ModuleMeta> next_modules;
        for (auto & entry : entries) {
            if (entry.module == module) {
                continue;
            }
            auto & meta = next_modules[entry.module];
            auto old_meta = modules.find(entry.module);
            if (old_meta != modules.end()) {
                meta.version_token = old_meta->second.version_token;
                meta.imports = old_meta->second.imports;
            }
            meta.rows.push_back(next.size());
            next_index.emplace(key(entry.module, entry.name), next.size());
            next.push_back(std::move(entry));
        }
        ModuleMeta meta;
        meta.version_token = token;
        meta.imports = imports;
        for (size_t i = 0; i < names.size(); ++i) {
            const std::string k = key(module, names[i]);
            if (next_index.emplace(k, next.size()).second) {
                meta.rows.push_back(next.size());
                next.push_back({module, names[i], embeddings[i]});
            }
        }
        next_modules[module] = std::move(meta);
        entries = std::move(next);
        index = std::move(next_index);
        modules = std::move(next_modules);
        n_dirty = entries.size();
    }

    void maybe_save(bool force) {
        std::lock_guard<std::mutex> lk(mu);
        if (n_dirty == 0 || vec_path.empty() || meta_path.empty()) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!force && n_dirty < SAVE_EVERY && now - last_save < std::chrono::seconds(SAVE_SECONDS)) {
            return;
        }

        std::vector<size_t> save_rows;
        save_rows.reserve(entries.size());
        for (const auto & item : modules) {
            for (size_t row : item.second.rows) {
                if (row < entries.size()) {
                    save_rows.push_back(row);
                }
            }
        }

        const std::string vec_tmp = vec_path + ".tmp";
        FILE * f = fopen(vec_tmp.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "premise cache: cannot write %s\n", vec_tmp.c_str());
            return;
        }
        int32_t header[2] = { (int32_t) save_rows.size(), (int32_t) dim };
        fwrite(header, sizeof(header), 1, f);
        for (size_t row : save_rows) {
            fwrite(entries[row].embedding.data(), sizeof(float), (size_t) dim, f);
        }
        fclose(f);

        const std::string meta_tmp = meta_path + ".tmp";
        std::ofstream fs(meta_tmp, std::ios::binary | std::ios::trunc);
        fs.write(META_MAGIC, sizeof(META_MAGIC));
        write_u32(fs, (uint32_t) modules.size());
        for (const auto & item : modules) {
            write_string(fs, item.first);
            write_string(fs, item.second.version_token);
            write_u32(fs, (uint32_t) item.second.imports.size());
            for (const auto & imported : item.second.imports) {
                write_string(fs, imported);
            }
            uint32_t n_rows = 0;
            for (size_t row : item.second.rows) {
                if (row < entries.size()) {
                    ++n_rows;
                }
            }
            write_u32(fs, n_rows);
            for (size_t row : item.second.rows) {
                if (row < entries.size()) {
                    write_string(fs, entries[row].name);
                }
            }
        }
        fs.close();
        if (!fs) {
            fprintf(stderr, "premise cache: cannot write %s\n", meta_tmp.c_str());
            std::remove(vec_tmp.c_str());
            return;
        }

        std::remove(vec_path.c_str()); // Windows rename does not overwrite
        std::rename(vec_tmp.c_str(), vec_path.c_str());
        std::remove(meta_path.c_str());
        std::rename(meta_tmp.c_str(), meta_path.c_str());

        fprintf(stderr, "premise cache: saved %zu embeddings\n", entries.size());
        n_dirty = 0;
        last_save = now;
    }
};
