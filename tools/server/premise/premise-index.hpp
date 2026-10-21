#pragma once

#include "premise-schema.hpp"

#include "ggml.h"

#include <sqlite3.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// The embeddings of every cached module's declarations: persisted in SQLite (as f16) and
// held in memory (as f32) for exact search. Not thread-safe; the server serializes requests.
struct premise_index {
    struct decl {
        std::string name;
        std::vector<float> embedding; // unit length
    };

    struct hit {
        std::string name;
        float score;
    };

    premise_index(const std::string & path, int dim) : dim(dim) {
        if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) != SQLITE_OK) {
            const std::string message = db ? sqlite3_errmsg(db) : "out of memory";
            sqlite3_close_v2(db);
            throw std::runtime_error("cannot open premise database " + path + ": " + message);
        }
        try {
            exec("PRAGMA foreign_keys = ON");
            init_schema();
            load();
        } catch (...) {
            sqlite3_close_v2(db);
            throw;
        }
    }

    ~premise_index() {
        sqlite3_close_v2(db);
    }

    premise_index(const premise_index &) = delete;
    premise_index & operator=(const premise_index &) = delete;

    // the module's version token, or "" if it is not cached
    std::string version(const std::string & module) const {
        auto it = modules.find(module);
        return it == modules.end() ? std::string() : it->second.token;
    }

    // make `decls` the module's declarations at `token`; empty and repeated names are dropped
    void replace(const std::string & module, const std::string & token, const std::vector<decl> & decls) {
        module_entry entry{token, {}, {}};
        std::unordered_set<std::string> seen;
        for (const auto & d : decls) {
            if (d.name.empty() || !seen.insert(d.name).second) {
                continue;
            }
            validate(d.embedding, "embedding");
            entry.names.push_back(d.name);
            entry.embeddings.insert(entry.embeddings.end(), d.embedding.begin(), d.embedding.end());
        }

        exec("BEGIN IMMEDIATE");
        try {
            statement remove(db, "DELETE FROM premise_modules WHERE module = ?1"); // cascades to declarations
            remove.bind(1, module);
            remove.run();
            statement insert_module(db, "INSERT INTO premise_modules(module, version_token) VALUES(?1, ?2)");
            insert_module.bind(1, module);
            insert_module.bind(2, token);
            insert_module.run();
            statement insert_decl(db, "INSERT INTO premise_declarations(module, ordinal, name) VALUES(?1, ?2, ?3)");
            statement insert_embedding(db,
                    "INSERT INTO premise_declaration_embeddings(declaration_id, embedding) VALUES(?1, ?2)");
            std::vector<ggml_fp16_t> half(dim);
            for (size_t i = 0; i < entry.names.size(); ++i) {
                insert_decl.bind(1, module);
                insert_decl.bind(2, (int64_t) i);
                insert_decl.bind(3, entry.names[i]);
                insert_decl.run();
                ggml_fp32_to_fp16_row(entry.embeddings.data() + i * dim, half.data(), dim);
                insert_embedding.bind(1, (int64_t) sqlite3_last_insert_rowid(db));
                insert_embedding.bind_blob(2, half.data(), (int) (half.size() * sizeof(ggml_fp16_t)));
                insert_embedding.run();
            }
            exec("COMMIT");
        } catch (...) {
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            throw;
        }
        modules[module] = std::move(entry);
    }

    // the k best-scoring declarations among `modules` and `locals`, one per name
    std::vector<hit> search(const std::vector<float> & query, const std::vector<std::string> & selected,
                            const std::vector<decl> & locals, int k) const {
        validate(query, "query embedding");
        if (k <= 0) {
            return {};
        }
        // keep 2k so that dropping repeated names (a lemma cached by several modules) still leaves k
        using scored = std::pair<float, const std::string *>;
        std::priority_queue<scored, std::vector<scored>, std::greater<scored>> best;
        auto offer = [&](float score, const std::string & name) {
            if ((int) best.size() < 2 * k) {
                best.emplace(score, &name);
            } else if (score > best.top().first) {
                best.pop();
                best.emplace(score, &name);
            }
        };
        for (const auto & module : selected) {
            auto it = modules.find(module);
            if (it == modules.end()) {
                continue;
            }
            const module_entry & entry = it->second;
            for (size_t i = 0; i < entry.names.size(); ++i) {
                offer(dot(query.data(), entry.embeddings.data() + i * dim), entry.names[i]);
            }
        }
        for (const auto & local : locals) {
            validate(local.embedding, "local embedding");
            offer(dot(query.data(), local.embedding.data()), local.name);
        }

        std::vector<scored> ranked;
        for (; !best.empty(); best.pop()) {
            ranked.push_back(best.top());
        }
        std::sort(ranked.begin(), ranked.end(), [](const scored & a, const scored & b) {
            return a.first != b.first ? a.first > b.first : *a.second < *b.second;
        });
        std::vector<hit> hits;
        std::unordered_set<std::string> seen;
        for (const auto & [score, name] : ranked) {
            if ((int) hits.size() < k && seen.insert(*name).second) {
                hits.push_back({*name, score});
            }
        }
        return hits;
    }

private:
    struct module_entry {
        std::string token;
        std::vector<std::string> names;
        std::vector<float> embeddings; // one row of `dim` per name
    };

    struct statement {
        sqlite3 * db;
        sqlite3_stmt * stmt = nullptr;

        statement(sqlite3 * db, const char * sql) : db(db) {
            check(sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr));
        }
        ~statement() {
            sqlite3_finalize(stmt);
        }

        void bind(int i, const std::string & text) {
            check(sqlite3_bind_text(stmt, i, text.data(), (int) text.size(), SQLITE_TRANSIENT));
        }
        void bind(int i, int64_t value) {
            check(sqlite3_bind_int64(stmt, i, value));
        }
        void bind_blob(int i, const void * data, int bytes) {
            check(sqlite3_bind_blob(stmt, i, data, bytes, SQLITE_TRANSIENT));
        }

        // true while there is a row to read
        bool step() {
            const int rc = sqlite3_step(stmt);
            if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
                check(rc);
            }
            return rc == SQLITE_ROW;
        }
        // execute a statement that returns no rows, ready to be bound again
        void run() {
            step();
            check(sqlite3_reset(stmt));
        }

        std::string text(int column) const {
            const char * data = reinterpret_cast<const char *>(sqlite3_column_text(stmt, column));
            return data ? std::string(data, (size_t) sqlite3_column_bytes(stmt, column)) : std::string();
        }

        void check(int rc) const {
            if (rc != SQLITE_OK) {
                throw std::runtime_error(std::string("SQLite: ") + sqlite3_errmsg(db));
            }
        }
    };

    sqlite3 * db = nullptr;
    int dim;
    std::unordered_map<std::string, module_entry> modules;

    void exec(const char * sql) {
        char * error = nullptr;
        if (sqlite3_exec(db, sql, nullptr, nullptr, &error) != SQLITE_OK) {
            const std::string message = error ? error : sqlite3_errmsg(db);
            sqlite3_free(error);
            throw std::runtime_error("SQLite: " + message);
        }
    }

    void init_schema() {
        exec("BEGIN IMMEDIATE");
        try {
            statement exists(db, "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'premise_config'");
            if (exists.step()) {
                statement config(db, "SELECT schema_version, embedding_dim FROM premise_config WHERE id = 1");
                if (!config.step() || sqlite3_column_int(config.stmt, 0) != premise_schema::VERSION) {
                    throw std::runtime_error("unsupported premise database schema version");
                }
                if (sqlite3_column_int(config.stmt, 1) != dim) {
                    throw std::runtime_error("premise database embedding dimension does not match the model");
                }
            } else {
                exec(premise_schema::SQL);
                statement config(db, "INSERT INTO premise_config(id, schema_version, embedding_dim) VALUES(1, ?1, ?2)");
                config.bind(1, (int64_t) premise_schema::VERSION);
                config.bind(2, (int64_t) dim);
                config.run();
            }
            exec("COMMIT");
        } catch (...) {
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            throw;
        }
    }

    void load() {
        statement tokens(db, "SELECT module, version_token FROM premise_modules");
        while (tokens.step()) {
            modules[tokens.text(0)].token = tokens.text(1);
        }
        statement rows(db,
                "SELECT d.module, d.name, e.embedding FROM premise_declarations d "
                "JOIN premise_declaration_embeddings e ON e.declaration_id = d.id ORDER BY d.module, d.ordinal");
        std::vector<ggml_fp16_t> half(dim);
        std::vector<float> row(dim);
        while (rows.step()) {
            if (sqlite3_column_bytes(rows.stmt, 2) != dim * (int) sizeof(ggml_fp16_t)) {
                throw std::runtime_error("stored embedding dimension mismatch");
            }
            std::memcpy(half.data(), sqlite3_column_blob(rows.stmt, 2), half.size() * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(half.data(), row.data(), dim);
            validate(row, "stored embedding");
            module_entry & entry = modules[rows.text(0)];
            entry.names.push_back(rows.text(1));
            entry.embeddings.insert(entry.embeddings.end(), row.begin(), row.end());
        }
    }

    void validate(const std::vector<float> & v, const char * what) const {
        if ((int) v.size() != dim) {
            throw std::runtime_error(std::string(what) + " has the wrong dimension");
        }
        double norm_squared = 0.0;
        for (float x : v) {
            norm_squared += (double) x * x;
        }
        if (!std::isfinite(norm_squared) || std::fabs(norm_squared - 1.0) > 1e-3) {
            throw std::runtime_error(std::string(what) + " is not unit length");
        }
    }

    // independent partial sums let compilers vectorize without reassociating
    float dot(const float * a, const float * b) const {
        float partial[8] = {};
        int i = 0;
        for (; i + 8 <= dim; i += 8) {
            for (int j = 0; j < 8; ++j) {
                partial[j] += a[i + j] * b[i + j];
            }
        }
        float sum = 0.0f;
        for (; i < dim; ++i) {
            sum += a[i] * b[i];
        }
        for (float p : partial) {
            sum += p;
        }
        return sum;
    }
};
