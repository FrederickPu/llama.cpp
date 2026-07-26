#pragma once

#include "json.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
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

// Owns premise metadata, embeddings, FAISS search, and optional persistence.
// Cache files contain a FAISS IndexFlatIP and aligned module/name metadata.
struct PremiseIndex {
    struct Record {
        std::string name;
        std::string decl;
        std::string module;
        std::vector<float> embedding;
    };

    struct Hit {
        std::string name;
        std::string statement;
        std::string decl;
        std::string module;
        float score = 0.0f;
    };

    struct ModuleMeta {
        std::string version_token;
        std::vector<std::string> imports;
        std::vector<size_t> rows;
    };

    int n_premises = 0;
    int dim = 0;

    void initialize_empty(int model_dim) {
        std::lock_guard<std::mutex> lock(mu);
        index_path.clear();
        metadata_path.clear();
        reset_locked(model_dim);
    }

    // Loads an externally generated global index. The JSON array contains the
    // declaration string aligned with each FAISS row.
    void load_offline(const char * faiss_path, const char * strings_path, int max_premises = 0) {
        std::lock_guard<std::mutex> lock(mu);
        index_path.clear();
        metadata_path.clear();

        std::ifstream input(strings_path);
        if (!input) {
            throw std::runtime_error(std::string("cannot open ") + strings_path);
        }
        nlohmann::json json;
        input >> json;
        std::vector<std::string> strings = json.get<std::vector<std::string>>();

        auto loaded_index = read_faiss_index(faiss_path);
        if (loaded_index->ntotal > std::numeric_limits<int>::max()) {
            throw std::runtime_error("premise index has too many rows");
        }

        dim = (int) loaded_index->d;
        const int total = (int) loaded_index->ntotal;
        n_premises = max_premises > 0 ? std::min(max_premises, total) : total;
        if ((int) strings.size() < n_premises) {
            throw std::runtime_error("premise metadata has fewer entries than FAISS index");
        }

        records.clear();
        records.reserve((size_t) n_premises);
        for (int i = 0; i < n_premises; ++i) {
            std::vector<float> embedding((size_t) dim);
            loaded_index->reconstruct(i, embedding.data());
            records.push_back({"", std::move(strings[(size_t) i]), "", std::move(embedding)});
        }
        modules.clear();

        if (n_premises == total) {
            faiss_index = std::move(loaded_index);
        } else {
            rebuild_faiss_index_locked();
        }
        dirty = false;
        unsaved_rows = 0;
        fprintf(stderr, "premise index: %d premises, dim=%d, backend=faiss\n", n_premises, dim);
    }

    // Opens the server-managed cache. Missing or invalid files produce an empty
    // cache; subsequent /cache requests repopulate and persist it.
    void load_cache(const std::string & faiss_path, const std::string & metadata_file, int model_dim) {
        std::lock_guard<std::mutex> lock(mu);
        index_path = faiss_path;
        metadata_path = metadata_file;
        reset_locked(model_dim);

        std::ifstream metadata(metadata_file, std::ios::binary);
        if (!metadata) {
            fprintf(stderr, "premise cache: starting empty (index files not present yet)\n");
            return;
        }

        try {
            std::unordered_map<std::string, ModuleMeta> loaded_modules;
            std::vector<std::pair<std::string, std::string>> loaded_names;
            read_metadata(metadata, loaded_modules, loaded_names);

            auto loaded_index = read_faiss_index(faiss_path.c_str());
            if ((int) loaded_index->d != model_dim) {
                throw std::runtime_error("embedding dimension does not match model");
            }

            const size_t index_rows = (size_t) loaded_index->ntotal;
            if (index_rows != loaded_names.size()) {
                throw std::runtime_error("FAISS rows do not match premise metadata");
            }
            std::unordered_set<std::string> seen;
            std::vector<Record> loaded_records;
            loaded_records.reserve(index_rows);
            for (auto & item : loaded_modules) {
                item.second.rows.clear();
            }
            for (size_t row = 0; row < index_rows; ++row) {
                const auto & module = loaded_names[row].first;
                const auto & name = loaded_names[row].second;
                if (!seen.insert(module + "\n" + name).second) {
                    throw std::runtime_error("duplicate declaration in premise metadata");
                }
                std::vector<float> embedding((size_t) model_dim);
                loaded_index->reconstruct((faiss::idx_t) row, embedding.data());
                loaded_modules[module].rows.push_back(loaded_records.size());
                loaded_records.push_back({name, "", module, std::move(embedding)});
            }

            records = std::move(loaded_records);
            modules = std::move(loaded_modules);
            n_premises = (int) records.size();
            faiss_index = std::move(loaded_index);
            fprintf(stderr, "premise cache: loaded %d embeddings from %s\n", n_premises, faiss_path.c_str());
        } catch (const std::exception & error) {
            reset_locked(model_dim);
            fprintf(stderr, "premise cache: ignoring %s (%s)\n", faiss_path.c_str(), error.what());
        }
    }

    std::string get_module_version(const std::string & module) const {
        std::lock_guard<std::mutex> lock(mu);
        auto it = modules.find(module);
        return it == modules.end() ? std::string() : it->second.version_token;
    }

    std::unordered_map<std::string, std::string> get_module_versions(
            const std::vector<std::string> & names) const {
        std::lock_guard<std::mutex> lock(mu);
        std::unordered_map<std::string, std::string> versions;
        for (const auto & module : names) {
            auto it = modules.find(module);
            if (it != modules.end()) {
                versions[module] = it->second.version_token;
            }
        }
        return versions;
    }

    // Atomically replaces every declaration owned by one module and rebuilds
    // row mappings and the FAISS index.
    void replace_module(const std::string & module,
                        const std::string & token,
                        const std::vector<std::string> & imports,
                        const std::vector<Record> & declarations) {
        std::lock_guard<std::mutex> lock(mu);
        int next_dim = dim;
        std::vector<Record> replacements;
        replacements.reserve(declarations.size());
        std::unordered_set<std::string> seen_names;
        for (auto declaration : declarations) {
            if (!declaration.name.empty() && !seen_names.insert(declaration.name).second) {
                continue;
            }
            if (next_dim == 0) {
                next_dim = (int) declaration.embedding.size();
            }
            if ((int) declaration.embedding.size() != next_dim) {
                throw std::runtime_error("premise embedding dimension mismatch");
            }
            declaration.module = module;
            replacements.push_back(std::move(declaration));
        }

        size_t removed_rows = 0;
        std::unordered_map<std::string, ModuleMeta> next_modules;
        for (const auto & item : modules) {
            if (item.first != module) {
                next_modules[item.first] = {item.second.version_token, item.second.imports, {}};
            } else {
                removed_rows = item.second.rows.size();
            }
        }

        std::vector<Record> next_records;
        next_records.reserve(records.size() + replacements.size());
        for (auto & record : records) {
            if (record.module == module) {
                continue;
            }
            next_modules[record.module].rows.push_back(next_records.size());
            next_records.push_back(std::move(record));
        }

        ModuleMeta replacement_meta{token, imports, {}};
        for (auto & replacement : replacements) {
            replacement_meta.rows.push_back(next_records.size());
            next_records.push_back(std::move(replacement));
        }
        next_modules[module] = std::move(replacement_meta);

        dim = next_dim;
        records = std::move(next_records);
        modules = std::move(next_modules);
        n_premises = (int) records.size();
        rebuild_faiss_index_locked();

        if (!index_path.empty()) {
            dirty = true;
            unsaved_rows += removed_rows + declarations.size();
        }
    }

    std::vector<Hit> search_all(const float * query, int top_k) const {
        std::lock_guard<std::mutex> lock(mu);
        const int count = std::min(top_k, n_premises);
        if (count <= 0 || !faiss_index) {
            return {};
        }

        std::vector<float> scores((size_t) count);
        std::vector<faiss::idx_t> labels((size_t) count);
        faiss_index->search(1, query, count, scores.data(), labels.data());

        std::vector<Hit> hits;
        hits.reserve((size_t) count);
        for (int i = 0; i < count; ++i) {
            if (labels[(size_t) i] >= 0 && labels[(size_t) i] < (faiss::idx_t) records.size()) {
                hits.push_back(make_hit_locked((size_t) labels[(size_t) i], scores[(size_t) i]));
            }
        }
        return hits;
    }

    // Restricts candidates to imported modules and request-local declarations.
    // This remains an explicit score pass because the candidate set is dynamic.
    std::vector<Hit> search_imports(const float * query,
                                    const std::vector<std::string> & imports,
                                    const std::vector<Record> & local,
                                    int top_k) const {
        struct ScoredHit {
            float score;
            Hit hit;
        };

        std::lock_guard<std::mutex> lock(mu);
        std::vector<size_t> rows;
        std::unordered_set<std::string> visited_modules;
        std::unordered_set<std::string> seen_names;
        for (const auto & module : imports) {
            collect_module_rows_locked(module, visited_modules, seen_names, rows);
        }

        std::vector<ScoredHit> scored;
        scored.reserve(rows.size() + local.size());
        for (size_t row : rows) {
            const float score = score_embedding(query, records[row].embedding);
            scored.push_back({score, make_hit_locked(row, score)});
        }
        for (const auto & record : local) {
            if (!record.name.empty() && !seen_names.insert(record.name).second) {
                continue;
            }
            if ((int) record.embedding.size() != dim) {
                continue;
            }
            const float score = score_embedding(query, record.embedding);
            scored.push_back({score, {record.name, record.decl, record.decl, record.module, score}});
        }

        const int count = std::min(top_k, (int) scored.size());
        if (count <= 0) {
            return {};
        }
        std::partial_sort(scored.begin(), scored.begin() + count, scored.end(),
                [](const ScoredHit & left, const ScoredHit & right) { return left.score > right.score; });

        std::vector<Hit> hits;
        hits.reserve((size_t) count);
        for (int i = 0; i < count; ++i) {
            hits.push_back(std::move(scored[(size_t) i].hit));
        }
        return hits;
    }

    // Persists the complete cache when requested after enough changes or time,
    // and unconditionally during shutdown.
    void flush(bool force) {
        std::lock_guard<std::mutex> lock(mu);
        if (!dirty || index_path.empty() || metadata_path.empty()) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!force && unsaved_rows < SAVE_EVERY && now - last_save < std::chrono::seconds(SAVE_SECONDS)) {
            return;
        }

        const std::string index_tmp = index_path + ".tmp";
        const std::string metadata_tmp = metadata_path + ".tmp";
        try {
            faiss::IndexFlatIP saved_index(dim);
            std::ofstream metadata(metadata_tmp, std::ios::binary | std::ios::trunc);
            if (!metadata) {
                throw std::runtime_error("cannot open metadata temporary file");
            }

            metadata.write(META_MAGIC, sizeof(META_MAGIC));
            write_u32(metadata, (uint32_t) modules.size());
            for (const auto & item : modules) {
                write_string(metadata, item.first);
                write_string(metadata, item.second.version_token);
                write_u32(metadata, (uint32_t) item.second.imports.size());
                for (const auto & imported : item.second.imports) {
                    write_string(metadata, imported);
                }

                uint32_t valid_rows = 0;
                for (size_t row : item.second.rows) {
                    valid_rows += row < records.size();
                }
                write_u32(metadata, valid_rows);
                for (size_t row : item.second.rows) {
                    if (row < records.size()) {
                        write_string(metadata, records[row].name);
                        saved_index.add(1, records[row].embedding.data());
                    }
                }
            }
            metadata.close();
            if (!metadata) {
                throw std::runtime_error("cannot write metadata temporary file");
            }
            faiss::write_index(&saved_index, index_tmp.c_str());
        } catch (const std::exception & error) {
            std::remove(index_tmp.c_str());
            std::remove(metadata_tmp.c_str());
            fprintf(stderr, "premise cache: save failed (%s)\n", error.what());
            return;
        }

        std::remove(index_path.c_str()); // Windows rename does not overwrite
        std::remove(metadata_path.c_str());
        if (std::rename(index_tmp.c_str(), index_path.c_str()) != 0 ||
                std::rename(metadata_tmp.c_str(), metadata_path.c_str()) != 0) {
            fprintf(stderr, "premise cache: failed to install saved files\n");
            return;
        }

        fprintf(stderr, "premise cache: saved %d embeddings\n", n_premises);
        dirty = false;
        unsaved_rows = 0;
        last_save = now;
    }

private:
    static constexpr size_t SAVE_EVERY = 1024;
    static constexpr int SAVE_SECONDS = 60;
    inline static constexpr char META_MAGIC[8] = {'P', 'M', 'N', 'A', 'M', 'E', '0', '1'};

    std::vector<Record> records;
    std::unordered_map<std::string, ModuleMeta> modules;
    std::unique_ptr<faiss::IndexFlat> faiss_index;
    std::string index_path;
    std::string metadata_path;
    bool dirty = false;
    size_t unsaved_rows = 0;
    std::chrono::steady_clock::time_point last_save = std::chrono::steady_clock::now();
    mutable std::mutex mu;

    static std::unique_ptr<faiss::IndexFlat> read_faiss_index(const char * path) {
        std::unique_ptr<faiss::Index> index(faiss::read_index(path));
        auto * flat = dynamic_cast<faiss::IndexFlat *>(index.get());
        if (!flat || flat->metric_type != faiss::METRIC_INNER_PRODUCT) {
            throw std::runtime_error(std::string("expected FAISS IndexFlatIP in ") + path);
        }
        index.release();
        return std::unique_ptr<faiss::IndexFlat>(flat);
    }

    void reset_locked(int model_dim) {
        n_premises = 0;
        dim = model_dim;
        records.clear();
        modules.clear();
        rebuild_faiss_index_locked();
        dirty = false;
        unsaved_rows = 0;
        last_save = std::chrono::steady_clock::now();
    }

    void rebuild_faiss_index_locked() {
        faiss_index = dim > 0 ? std::make_unique<faiss::IndexFlatIP>(dim) : nullptr;
        if (!faiss_index) {
            return;
        }
        for (const auto & record : records) {
            if ((int) record.embedding.size() != dim) {
                throw std::runtime_error("premise embedding dimension mismatch");
            }
            faiss_index->add(1, record.embedding.data());
        }
    }

    float score_embedding(const float * query, const std::vector<float> & embedding) const {
        float score = 0.0f;
        for (int i = 0; i < dim; ++i) {
            score += query[i] * embedding[(size_t) i];
        }
        return score;
    }

    Hit make_hit_locked(size_t row, float score) const {
        const auto & record = records[row];
        return {record.name, record.decl, record.decl, record.module, score};
    }

    void collect_module_rows_locked(const std::string & module,
                                    std::unordered_set<std::string> & visited_modules,
                                    std::unordered_set<std::string> & seen_names,
                                    std::vector<size_t> & rows) const {
        if (!visited_modules.insert(module).second) {
            return;
        }
        auto it = modules.find(module);
        if (it == modules.end()) {
            return;
        }
        for (const auto & imported : it->second.imports) {
            collect_module_rows_locked(imported, visited_modules, seen_names, rows);
        }
        for (size_t row : it->second.rows) {
            if (row < records.size() && (records[row].name.empty() || seen_names.insert(records[row].name).second)) {
                rows.push_back(row);
            }
        }
    }

    static bool read_u32(std::istream & input, uint32_t & value) {
        input.read(reinterpret_cast<char *>(&value), sizeof(value));
        return !!input;
    }

    static bool read_string(std::istream & input, std::string & value) {
        uint32_t size = 0;
        if (!read_u32(input, size)) {
            return false;
        }
        value.resize(size);
        if (size > 0) {
            input.read(value.data(), size);
        }
        return !!input;
    }

    static void read_metadata(std::istream & input,
                              std::unordered_map<std::string, ModuleMeta> & loaded_modules,
                              std::vector<std::pair<std::string, std::string>> & loaded_names) {
        char magic[sizeof(META_MAGIC)] = {};
        input.read(magic, sizeof(magic));
        if (!input || std::memcmp(magic, META_MAGIC, sizeof(META_MAGIC)) != 0) {
            throw std::runtime_error("unsupported premise metadata format");
        }

        uint32_t module_count = 0;
        if (!read_u32(input, module_count)) {
            throw std::runtime_error("truncated premise metadata");
        }
        for (uint32_t i = 0; i < module_count; ++i) {
            std::string module;
            ModuleMeta metadata;
            if (!read_string(input, module) || !read_string(input, metadata.version_token)) {
                throw std::runtime_error("truncated premise metadata");
            }

            uint32_t import_count = 0;
            if (!read_u32(input, import_count)) {
                throw std::runtime_error("truncated premise metadata");
            }
            metadata.imports.reserve(import_count);
            for (uint32_t j = 0; j < import_count; ++j) {
                std::string imported;
                if (!read_string(input, imported)) {
                    throw std::runtime_error("truncated premise metadata");
                }
                metadata.imports.push_back(std::move(imported));
            }

            uint32_t declaration_count = 0;
            if (!read_u32(input, declaration_count)) {
                throw std::runtime_error("truncated premise metadata");
            }
            loaded_modules[module] = std::move(metadata);
            for (uint32_t j = 0; j < declaration_count; ++j) {
                std::string name;
                if (!read_string(input, name)) {
                    throw std::runtime_error("truncated premise metadata");
                }
                loaded_names.push_back({module, std::move(name)});
            }
        }
    }

    static void write_u32(std::ostream & output, uint32_t value) {
        output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    static void write_string(std::ostream & output, const std::string & value) {
        write_u32(output, (uint32_t) value.size());
        output.write(value.data(), (std::streamsize) value.size());
    }
};
