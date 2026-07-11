#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
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
