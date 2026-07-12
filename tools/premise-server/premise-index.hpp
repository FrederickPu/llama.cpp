#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "json.hpp"

struct PremiseIndex {
    int n_premises = 0;
    int dim        = 0;
    std::vector<float>       vectors;  // [n_premises * dim], row-major, L2-normalised
    std::vector<std::string> strings;

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
            throw std::runtime_error("premise_strings.json has fewer entries than premise_vectors.bin");
        }
        fprintf(stderr, "premise index: %d premises, dim=%d\n", n_premises, dim);
    }

    // Brute-force cosine similarity (inner product on L2-normalised vectors).
    // Returns top_k (premise_string, score) pairs, sorted by descending score.
    std::vector<std::pair<std::string, float>>
    search(const float * query, int top_k) const {
        std::vector<std::pair<float, int>> scores(n_premises);
        for (int i = 0; i < n_premises; ++i) {
            float dot = 0.0f;
            const float * row = vectors.data() + (size_t)i * dim;
            for (int d = 0; d < dim; ++d) dot += query[d] * row[d];
            scores[i] = {dot, i};
        }
        int k = std::min(top_k, n_premises);
        std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                          [](const auto & a, const auto & b){ return a.first > b.first; });
        std::vector<std::pair<std::string, float>> out;
        out.reserve(k);
        for (int i = 0; i < k; ++i)
            out.push_back({strings[scores[i].second], scores[i].first});
        return out;
    }
};

// Persistent decl-string -> embedding cache used by the standalone
// premise-server (PremiseMode::Embedding), so restarts don't re-embed every
// declaration sent to /cache. Stored in the same two-file layout as
// PremiseIndex (JSON array of strings + packed float rows), so the files it
// writes can also serve as an offline joint-server index. Missing or invalid
// files mean an empty cache; the server creates and appends to them itself.
struct PremiseEmbedCache {
    static constexpr size_t SAVE_EVERY   = 1024; // unsaved rows that force a save
    static constexpr int    SAVE_SECONDS = 60;   // dirty-cache flush interval

    std::string vec_path;
    std::string str_path;
    int dim = 0;

    std::mutex mu;
    std::vector<std::string>                 strings; // insertion order == row order
    std::vector<std::vector<float>>          vectors;
    std::unordered_map<std::string, size_t>  index;   // decl string -> row
    size_t n_saved = 0;                               // leading rows already on disk
    std::chrono::steady_clock::time_point last_save = std::chrono::steady_clock::now();

    void load(const std::string & vec, const std::string & str, int model_dim) {
        vec_path = vec;
        str_path = str;
        dim      = model_dim;
        std::ifstream fs(str_path);
        std::ifstream fv(vec_path, std::ios::binary);
        if (!fs || !fv) {
            fprintf(stderr, "premise cache: starting empty (index files not present yet)\n");
            return;
        }
        std::vector<std::string> loaded;
        try {
            nlohmann::json j; fs >> j;
            loaded = j.get<std::vector<std::string>>();
        } catch (const std::exception & e) {
            fprintf(stderr, "premise cache: ignoring unreadable %s (%s)\n", str_path.c_str(), e.what());
            return;
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
            if (index.emplace(loaded[i], strings.size()).second) {
                strings.push_back(loaded[i]);
                vectors.push_back(std::move(row));
            }
        }
        n_saved = strings.size();
        // If truncation or dedup changed the row set, rewrite from scratch on
        // the next save instead of appending to inconsistent files.
        if (n_saved != loaded.size() || (int32_t) count != n) {
            n_saved = 0;
        }
        fprintf(stderr, "premise cache: loaded %zu embeddings from %s\n", strings.size(), vec_path.c_str());
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return strings.size();
    }

    bool find(const std::string & decl, std::vector<float> & out) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = index.find(decl);
        if (it == index.end()) {
            return false;
        }
        out = vectors[it->second];
        return true;
    }

    void insert(const std::string & decl, const std::vector<float> & embedding) {
        std::lock_guard<std::mutex> lk(mu);
        if (index.emplace(decl, strings.size()).second) {
            strings.push_back(decl);
            vectors.push_back(embedding);
        }
    }

    void maybe_save(bool force) {
        std::lock_guard<std::mutex> lk(mu);
        const size_t unsaved = strings.size() - n_saved;
        if (unsaved == 0 || vec_path.empty()) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!force && unsaved < SAVE_EVERY && now - last_save < std::chrono::seconds(SAVE_SECONDS)) {
            return;
        }

        FILE * f = fopen(vec_path.c_str(), n_saved == 0 ? "wb" : "r+b");
        if (!f) {
            fprintf(stderr, "premise cache: cannot write %s\n", vec_path.c_str());
            return;
        }
        int32_t header[2] = { (int32_t) n_saved, (int32_t) dim };
        if (n_saved == 0) {
            fwrite(header, sizeof(header), 1, f);
        }
        fseek(f, (long) (sizeof(header) + n_saved * (size_t) dim * sizeof(float)), SEEK_SET);
        for (size_t i = n_saved; i < vectors.size(); ++i) {
            fwrite(vectors[i].data(), sizeof(float), (size_t) dim, f);
        }
        // Patch the row count last, so a torn write leaves a shorter-but-valid file.
        header[0] = (int32_t) vectors.size();
        fseek(f, 0, SEEK_SET);
        fwrite(header, sizeof(header), 1, f);
        fclose(f);

        const std::string tmp = str_path + ".tmp";
        std::ofstream fs(tmp, std::ios::trunc);
        fs << nlohmann::json(strings);
        fs.close();
        if (fs) {
            std::remove(str_path.c_str()); // Windows rename does not overwrite
            std::rename(tmp.c_str(), str_path.c_str());
        }

        fprintf(stderr, "premise cache: saved %zu embeddings (%zu new)\n", strings.size(), unsaved);
        n_saved = strings.size();
        last_save = now;
    }
};
