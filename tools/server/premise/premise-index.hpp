#pragma once

#include "premise-schema.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>
#include <sqlite3.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct PremiseIndex {
    static_assert(sizeof(faiss::idx_t) == sizeof(int64_t));

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

    ~PremiseIndex() {
        if (db) {
            sqlite3_close_v2(db);
        }
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu);
        require_db();
        refresh_database_locked();
        return rows.size();
    }

    int embedding_dim() const {
        std::lock_guard<std::mutex> lock(mu);
        require_db();
        refresh_database_locked();
        return dim;
    }

    bool find_local_embedding(const std::string & text, std::vector<float> & embedding) {
        std::lock_guard<std::mutex> lock(local_cache_mu);
        auto found = local_cache_index.find(text);
        if (found == local_cache_index.end()) {
            return false;
        }
        local_cache.splice(local_cache.begin(), local_cache, found->second);
        embedding = found->second->second;
        return true;
    }

    void cache_local_embedding(const std::string & text, const std::vector<float> & embedding) {
        std::lock_guard<std::mutex> lock(local_cache_mu);
        auto found = local_cache_index.find(text);
        if (found != local_cache_index.end()) {
            local_cache.splice(local_cache.begin(), local_cache, found->second);
            found->second->second = embedding;
            return;
        }
        local_cache.emplace_front(text, embedding);
        local_cache_index.emplace(local_cache.front().first, local_cache.begin());
        if (local_cache.size() > LOCAL_CACHE_CAPACITY) {
            local_cache_index.erase(local_cache.back().first);
            local_cache.pop_back();
        }
    }

    void open(const std::string & path, int model_dim) {
        std::lock_guard<std::mutex> lock(mu);
        if (db) {
            throw std::runtime_error("premise database is already open");
        }
        if (path.empty()) {
            throw std::runtime_error("missing --index-db");
        }
        if (model_dim <= 0) {
            throw std::runtime_error("invalid embedding dimension");
        }
        if (model_dim > std::numeric_limits<int>::max() / (int) sizeof(float)) {
            throw std::runtime_error("embedding dimension is too large");
        }
        sqlite3 * opened = nullptr;
        const int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX;
        const int rc = sqlite3_open_v2(path.c_str(), &opened, flags, nullptr);
        if (rc != SQLITE_OK) {
            const std::string message = opened ? sqlite3_errmsg(opened) : "cannot allocate SQLite connection";
            if (opened) {
                sqlite3_close_v2(opened);
            }
            throw std::runtime_error("cannot open premise database: " + message);
        }

        db = opened;
        sidecar_path = path + ".faiss";
        dim = model_dim;
        sqlite3_extended_result_codes(db, 1);
        sqlite3_busy_timeout(db, 5000);

        try {
            exec(db, "PRAGMA foreign_keys = ON");
            exec(db, "PRAGMA journal_mode = WAL");
            exec(db, "PRAGMA synchronous = FULL");
            exec(db, "PRAGMA wal_autocheckpoint = 1000");
            initialize_schema_locked();
            refresh_database_locked(true);
        } catch (...) {
            sqlite3_close_v2(db);
            db = nullptr;
            sidecar_path.clear();
            throw;
        }
    }

    std::string get_module_version(const std::string & module) const {
        std::lock_guard<std::mutex> lock(mu);
        require_db();
        refresh_database_locked();
        auto it = modules.find(module);
        return it == modules.end() ? std::string() : it->second.version_token;
    }

    void replace_module(const std::string & module,
                        const std::string & version_token,
                        const std::vector<std::string> & imports,
                        const std::vector<Candidate> & replacements) {
        std::lock_guard<std::mutex> lock(mu);
        require_db();
        refresh_database_locked();

        std::vector<const Candidate *> accepted;
        accepted.reserve(replacements.size());
        std::unordered_set<std::string> seen;
        for (const auto & candidate : replacements) {
            if (candidate.name.empty() || !seen.insert(candidate.name).second) {
                continue;
            }
            validate_vector(candidate.embedding, dim, "embedding");
            accepted.push_back(&candidate);
        }

        std::vector<faiss::idx_t> new_ids;
        for (;;) {
            exec(db, "BEGIN IMMEDIATE");
            try {
                if (read_content_revision(db) != content_revision) {
                    exec(db, "ROLLBACK");
                    refresh_database_locked(true);
                    continue;
                }
                if (content_revision == std::numeric_limits<int64_t>::max()) {
                    throw std::runtime_error("premise database content revision is exhausted");
                }
                next_faiss_id = std::max(next_faiss_id, read_faiss_next_id(database_identity));

                Statement delete_module(db, "DELETE FROM premise_modules WHERE module = ?1");
                delete_module.bind_text(1, module);
                delete_module.run();

                Statement insert_module(db,
                        "INSERT INTO premise_modules(module, version_token) VALUES(?1, ?2)");
                insert_module.bind_text(1, module);
                insert_module.bind_text(2, version_token);
                insert_module.run();

                Statement insert_import(db,
                        "INSERT INTO premise_imports(module, ordinal, imported_module) VALUES(?1, ?2, ?3)");
                for (size_t i = 0; i < imports.size(); ++i) {
                    insert_import.bind_text(1, module);
                    insert_import.bind_int64(2, checked_int64(i, "too many module imports"));
                    insert_import.bind_text(3, imports[i]);
                    insert_import.run();
                    insert_import.reset();
                }

                Statement insert_declaration(db,
                        "INSERT INTO premise_declarations(id, module, ordinal, name) VALUES(?1, ?2, ?3, ?4)");
                new_ids.clear();
                new_ids.reserve(accepted.size());
                std::vector<float> embeddings;
                embeddings.reserve(accepted.size() * (size_t) dim);
                uint64_t candidate_next_id = next_faiss_id;
                for (size_t i = 0; i < accepted.size(); ++i) {
                    if (candidate_next_id > (uint64_t) std::numeric_limits<faiss::idx_t>::max()) {
                        throw std::runtime_error("premise FAISS declaration IDs are exhausted");
                    }
                    const faiss::idx_t id = (faiss::idx_t) candidate_next_id++;
                    insert_declaration.bind_int64(1, id);
                    insert_declaration.bind_text(2, module);
                    insert_declaration.bind_int64(3, checked_int64(i, "too many declarations"));
                    insert_declaration.bind_text(4, accepted[i]->name);
                    insert_declaration.run();
                    insert_declaration.reset();
                    new_ids.push_back(id);
                    embeddings.insert(embeddings.end(),
                            accepted[i]->embedding.begin(), accepted[i]->embedding.end());
                }

                if (!new_ids.empty()) {
                    base_index->add_with_ids(
                            (faiss::idx_t) new_ids.size(), embeddings.data(), new_ids.data());
                    base_ids.insert(new_ids.begin(), new_ids.end());
                }
                persist_faiss_index(*base_index, base_ids, database_identity, candidate_next_id);
                next_faiss_id = candidate_next_id;

                Statement update_revision(db,
                        "UPDATE premise_config SET content_revision = content_revision + 1 WHERE id = 1");
                update_revision.run();
                if (sqlite3_changes(db) != 1) {
                    throw std::runtime_error("premise database has no configuration row");
                }
                exec(db, "COMMIT");
                break;
            } catch (...) {
                rollback(db);
                try {
                    refresh_database_locked(true);
                } catch (...) {
                }
                throw;
            }
        }

        try {
            apply_committed_replacement_locked(module, version_token, imports, accepted, new_ids);
        } catch (...) {
            refresh_database_locked(true);
        }
    }

    std::vector<Hit> search_global(const float * query, int top_k) const {
        std::lock_guard<std::mutex> lock(mu);
        require_db();
        refresh_database_locked();
        if (top_k <= 0) {
            return {};
        }
        validate_vector(query, dim, "query embedding");

        std::vector<Hit> hits = search_base(query, top_k, nullptr);
        trim_hits(hits, top_k);
        return hits;
    }

    std::vector<Hit> search_scoped(const float * query,
                                   const std::vector<std::string> & import_roots,
                                   const std::vector<Candidate> & locals,
                                   int top_k) const {
        std::lock_guard<std::mutex> lock(mu);
        require_db();
        refresh_database_locked();
        if (top_k <= 0) {
            return {};
        }
        validate_vector(query, dim, "query embedding");

        if (scope_cache.revision != content_revision || scope_cache.import_roots != import_roots) {
            scope_cache = {};
            scope_cache.revision = content_revision;
            scope_cache.import_roots = import_roots;
            std::unordered_set<std::string> visited;
            for (const auto & root : import_roots) {
                collect_candidate_rows(root, visited, scope_cache.names, scope_cache.ids);
            }
        }

        std::vector<Hit> hits = search_base(query, top_k, &scope_cache.ids);
        score_locals(query, locals, &scope_cache.names, hits);
        trim_hits(hits, top_k);
        return hits;
    }

    bool loaded_sidecar_for_test() const {
        std::lock_guard<std::mutex> lock(mu);
        require_db();
        return sidecar_loaded;
    }

private:
    using LocalCache = std::list<std::pair<std::string, std::vector<float>>>;
    static constexpr size_t LOCAL_CACHE_CAPACITY = 4096;

    struct Row {
        std::string name;
        std::string module;
    };

    struct DatabaseState {
        std::unordered_map<faiss::idx_t, Row> rows;
        std::unordered_map<std::string, ModuleEntry> modules;
        std::array<uint8_t, 16> identity = {};
        int64_t revision = 0;
    };

    struct FaissCandidate {
        std::unique_ptr<faiss::IndexIDMap2> index;
        std::unordered_set<faiss::idx_t> ids;
        std::array<uint8_t, 16> identity = {};
        uint64_t next_id = 1;
        bool loaded = false;
    };

    struct ScopeCache {
        int64_t revision = -1;
        std::vector<std::string> import_roots;
        std::vector<faiss::idx_t> ids;
        std::unordered_set<std::string> names;
    };

    struct Statement {
        sqlite3 * db = nullptr;
        sqlite3_stmt * stmt = nullptr;

        Statement(sqlite3 * database, const char * sql) : db(database) {
            const int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
            if (rc != SQLITE_OK) {
                throw std::runtime_error(std::string("SQLite prepare failed: ") + sqlite3_errmsg(db));
            }
        }

        ~Statement() {
            sqlite3_finalize(stmt);
        }

        void bind_int(int index, int value) {
            check(sqlite3_bind_int(stmt, index, value));
        }

        void bind_int64(int index, int64_t value) {
            check(sqlite3_bind_int64(stmt, index, value));
        }

        void bind_text(int index, const std::string & value) {
            if (value.size() > (size_t) std::numeric_limits<int>::max()) {
                throw std::runtime_error("text value is too large for SQLite");
            }
            check(sqlite3_bind_text(stmt, index, value.data(), (int) value.size(), SQLITE_TRANSIENT));
        }

        bool step() {
            const int rc = sqlite3_step(stmt);
            if (rc == SQLITE_ROW) {
                return true;
            }
            if (rc != SQLITE_DONE) {
                throw std::runtime_error(std::string("SQLite step failed: ") + sqlite3_errmsg(db));
            }
            return false;
        }

        void run() {
            if (step()) {
                throw std::runtime_error("SQLite statement unexpectedly returned a row");
            }
        }

        void reset() {
            check(sqlite3_reset(stmt));
            check(sqlite3_clear_bindings(stmt));
        }

    private:
        void check(int rc) {
            if (rc != SQLITE_OK) {
                throw std::runtime_error(std::string("SQLite operation failed: ") + sqlite3_errmsg(db));
            }
        }
    };

    sqlite3 * db = nullptr;
    std::string sidecar_path;
    mutable std::array<uint8_t, 16> database_identity = {};
    mutable uint64_t next_faiss_id = 1;
    int dim = 0;
    mutable int observed_data_version = -1;
    mutable int64_t content_revision = 0;
    mutable std::unordered_map<faiss::idx_t, Row> rows;
    mutable std::unordered_map<std::string, ModuleEntry> modules;
    mutable std::unique_ptr<faiss::IndexIDMap2> base_index;
    mutable std::unordered_set<faiss::idx_t> base_ids;
    mutable ScopeCache scope_cache;
    std::mutex local_cache_mu;
    LocalCache local_cache;
    std::unordered_map<std::string, LocalCache::iterator> local_cache_index;
    mutable std::mutex mu;
    mutable bool sidecar_loaded = false;

    void require_db() const {
        if (!db) {
            throw std::runtime_error("premise database is not open");
        }
    }

    static void exec(sqlite3 * database, const char * sql) {
        char * error = nullptr;
        const int rc = sqlite3_exec(database, sql, nullptr, nullptr, &error);
        if (rc != SQLITE_OK) {
            const std::string message = error ? error : sqlite3_errmsg(database);
            sqlite3_free(error);
            throw std::runtime_error("SQLite execution failed: " + message);
        }
    }

    static void rollback(sqlite3 * database) noexcept {
        sqlite3_exec(database, "ROLLBACK", nullptr, nullptr, nullptr);
    }

    static int64_t checked_int64(size_t value, const char * message) {
        if (value > (size_t) std::numeric_limits<int64_t>::max()) {
            throw std::runtime_error(message);
        }
        return (int64_t) value;
    }

    static std::string column_text(sqlite3_stmt * statement, int column) {
        const char * data = reinterpret_cast<const char *>(sqlite3_column_text(statement, column));
        const int bytes = sqlite3_column_bytes(statement, column);
        return data ? std::string(data, (size_t) bytes) : std::string();
    }

    static void validate_vector(const float * values, int dimensions, const char * what) {
        if (!values || dimensions <= 0) {
            throw std::runtime_error(std::string(what) + " is empty");
        }
        double norm_squared = 0.0;
        for (int i = 0; i < dimensions; ++i) {
            if (!std::isfinite(values[i])) {
                throw std::runtime_error(std::string(what) + " contains a non-finite component");
            }
            norm_squared += (double) values[i] * values[i];
        }
        if (!std::isfinite(norm_squared) || std::fabs(norm_squared - 1.0) > 1e-3) {
            throw std::runtime_error(std::string(what) + " is not approximately unit normalized");
        }
    }

    static void validate_vector(const std::vector<float> & values, int dimensions, const char * what) {
        if ((int) values.size() != dimensions) {
            throw std::runtime_error(std::string(what) + " dimension mismatch");
        }
        validate_vector(values.data(), dimensions, what);
    }

    static bool table_exists(sqlite3 * database, const char * name) {
        Statement statement(database,
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?1");
        statement.bind_text(1, name);
        return statement.step();
    }

    static int64_t read_content_revision(sqlite3 * database) {
        Statement statement(database,
                "SELECT content_revision FROM premise_config WHERE id = 1");
        if (!statement.step()) {
            throw std::runtime_error("premise database has no configuration row");
        }
        if (sqlite3_column_type(statement.stmt, 0) != SQLITE_INTEGER) {
            throw std::runtime_error("invalid premise database content revision");
        }
        const int64_t revision = sqlite3_column_int64(statement.stmt, 0);
        if (revision < 0 || statement.step()) {
            throw std::runtime_error("invalid premise database content revision");
        }
        return revision;
    }

    int data_version_locked() const {
        Statement statement(db, "PRAGMA data_version");
        if (!statement.step()) {
            throw std::runtime_error("SQLite did not return a data version");
        }
        return sqlite3_column_int(statement.stmt, 0);
    }

    void initialize_schema_locked() {
        exec(db, "BEGIN IMMEDIATE");
        try {
            if (table_exists(db, "premise_config")) {
                Statement read_config(db,
                        "SELECT schema_version, embedding_dim, database_identity "
                        "FROM premise_config WHERE id = 1");
                if (!read_config.step()) {
                    throw std::runtime_error("premise database has no configuration row");
                }
                const int stored_schema = sqlite3_column_int(read_config.stmt, 0);
                const int stored_dim = sqlite3_column_int(read_config.stmt, 1);
                const bool valid_identity = sqlite3_column_type(read_config.stmt, 2) == SQLITE_BLOB &&
                        sqlite3_column_bytes(read_config.stmt, 2) == 16;
                if (read_config.step()) {
                    throw std::runtime_error("premise database has multiple configuration rows");
                }
                if (stored_schema != premise_schema::VERSION) {
                    throw std::runtime_error("unsupported premise database schema version");
                }
                if (stored_dim != dim) {
                    throw std::runtime_error("premise database embedding dimension does not match model");
                }
                if (!valid_identity) {
                    throw std::runtime_error("invalid premise database identity");
                }
                exec(db, premise_schema::SQL);
            } else {
                exec(db, premise_schema::SQL);
                Statement insert_config(db,
                        "INSERT INTO premise_config("
                        "id, schema_version, embedding_dim, content_revision, database_identity) "
                        "VALUES(1, ?1, ?2, 0, randomblob(16))");
                insert_config.bind_int(1, premise_schema::VERSION);
                insert_config.bind_int(2, dim);
                insert_config.run();
            }
            Statement declaration_schema(db,
                    "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'premise_declarations'");
            if (!declaration_schema.step() ||
                    sqlite3_column_type(declaration_schema.stmt, 0) != SQLITE_TEXT) {
                throw std::runtime_error("premise declaration schema is missing");
            }
            std::string declaration_sql = column_text(declaration_schema.stmt, 0);
            std::transform(declaration_sql.begin(), declaration_sql.end(), declaration_sql.begin(),
                    [](unsigned char c) { return (char) std::toupper(c); });
            if (declaration_sql.find("AUTOINCREMENT") == std::string::npos || declaration_schema.step()) {
                throw std::runtime_error("premise declaration IDs must not be reused");
            }
            exec(db, "COMMIT");
        } catch (...) {
            rollback(db);
            throw;
        }
    }

    inline static constexpr char SIDECAR_MAGIC[8] = {'P', 'M', 'F', 'A', 'I', 'S', '0', '2'};
    inline static constexpr uint32_t SIDECAR_VERSION = 2;

    static DatabaseState read_database_state(sqlite3 * database, int dimensions) {
        DatabaseState state;
        {
            Statement config(database,
                    "SELECT schema_version, embedding_dim, content_revision, database_identity "
                    "FROM premise_config WHERE id = 1");
            if (!config.step() ||
                    sqlite3_column_type(config.stmt, 0) != SQLITE_INTEGER ||
                    sqlite3_column_type(config.stmt, 1) != SQLITE_INTEGER ||
                    sqlite3_column_type(config.stmt, 2) != SQLITE_INTEGER ||
                    sqlite3_column_type(config.stmt, 3) != SQLITE_BLOB ||
                    sqlite3_column_int(config.stmt, 0) != premise_schema::VERSION ||
                    sqlite3_column_int(config.stmt, 1) != dimensions ||
                    sqlite3_column_bytes(config.stmt, 3) != (int) state.identity.size()) {
                throw std::runtime_error("invalid premise database configuration");
            }
            state.revision = sqlite3_column_int64(config.stmt, 2);
            std::memcpy(state.identity.data(), sqlite3_column_blob(config.stmt, 3), state.identity.size());
            if (state.revision < 0 || config.step()) {
                throw std::runtime_error("invalid premise database configuration");
            }
        }

        Statement read_modules(database,
                "SELECT module, version_token FROM premise_modules ORDER BY module");
        while (read_modules.step()) {
            const std::string module = column_text(read_modules.stmt, 0);
            const std::string token = column_text(read_modules.stmt, 1);
            if (!state.modules.emplace(module, ModuleEntry{token, {}, {}}).second) {
                throw std::runtime_error("duplicate module metadata in premise database");
            }
        }

        Statement read_imports(database,
                "SELECT module, imported_module FROM premise_imports ORDER BY module, ordinal");
        while (read_imports.step()) {
            const std::string module = column_text(read_imports.stmt, 0);
            auto it = state.modules.find(module);
            if (it == state.modules.end()) {
                throw std::runtime_error("premise import references an unknown module");
            }
            it->second.imports.push_back(column_text(read_imports.stmt, 1));
        }

        Statement read_declarations(database,
                "SELECT id, module, name FROM premise_declarations ORDER BY module, ordinal");
        while (read_declarations.step()) {
            const faiss::idx_t id = sqlite3_column_int64(read_declarations.stmt, 0);
            const std::string module = column_text(read_declarations.stmt, 1);
            const std::string name = column_text(read_declarations.stmt, 2);
            auto module_it = state.modules.find(module);
            if (id <= 0 || name.empty() || module_it == state.modules.end()) {
                throw std::runtime_error("invalid declaration metadata in premise database");
            }
            if (!state.rows.emplace(id, Row{name, module}).second) {
                throw std::runtime_error("duplicate declaration ID in premise database");
            }
            module_it->second.declaration_ids.push_back(id);
        }
        return state;
    }

    static std::unique_ptr<faiss::IndexIDMap2> make_index(int dimensions) {
        auto result = std::make_unique<faiss::IndexIDMap2>(new faiss::IndexFlatIP(dimensions));
        result->own_fields = true;
        return result;
    }

    static void read_exact(FILE * file, void * data, size_t size) {
        if (size && std::fread(data, 1, size, file) != size) {
            throw std::runtime_error("truncated premise FAISS sidecar");
        }
    }

    static uint32_t read_u32(FILE * file) {
        uint8_t data[4];
        read_exact(file, data, sizeof(data));
        return (uint32_t) data[0] |
                (uint32_t) data[1] << 8 |
                (uint32_t) data[2] << 16 |
                (uint32_t) data[3] << 24;
    }

    static uint64_t read_u64(FILE * file) {
        uint8_t data[8];
        read_exact(file, data, sizeof(data));
        uint64_t value = 0;
        for (int i = 7; i >= 0; --i) {
            value = (value << 8) | data[i];
        }
        return value;
    }

    static void write_exact(FILE * file, const void * data, size_t size) {
        if (size && std::fwrite(data, 1, size, file) != size) {
            throw std::runtime_error("cannot write premise FAISS sidecar");
        }
    }

    static void write_u32(FILE * file, uint32_t value) {
        uint8_t data[4];
        for (size_t i = 0; i < sizeof(data); ++i) {
            data[i] = (uint8_t) (value >> (i * 8));
        }
        write_exact(file, data, sizeof(data));
    }

    static void write_u64(FILE * file, uint64_t value) {
        uint8_t data[8];
        for (size_t i = 0; i < sizeof(data); ++i) {
            data[i] = (uint8_t) (value >> (i * 8));
        }
        write_exact(file, data, sizeof(data));
    }

    uint64_t read_faiss_next_id(const std::array<uint8_t, 16> & expected_identity) const {
        FILE * file = std::fopen(sidecar_path.c_str(), "rb");
        if (!file) {
            throw std::runtime_error("cannot open premise FAISS sidecar");
        }
        try {
            char magic[sizeof(SIDECAR_MAGIC)];
            read_exact(file, magic, sizeof(magic));
            if (std::memcmp(magic, SIDECAR_MAGIC, sizeof(magic)) != 0 ||
                    read_u32(file) != SIDECAR_VERSION ||
                    read_u32(file) != (uint32_t) premise_schema::VERSION) {
                throw std::runtime_error("invalid premise FAISS sidecar header");
            }
            std::array<uint8_t, 16> identity;
            read_exact(file, identity.data(), identity.size());
            const uint64_t next_id = read_u64(file);
            if (identity != expected_identity || next_id == 0 ||
                    next_id > (uint64_t) std::numeric_limits<faiss::idx_t>::max() + 1ULL) {
                throw std::runtime_error("premise FAISS sidecar does not match database");
            }
            if (std::fclose(file) != 0) {
                file = nullptr;
                throw std::runtime_error("cannot close premise FAISS sidecar");
            }
            return next_id;
        } catch (...) {
            if (file) {
                std::fclose(file);
            }
            throw;
        }
    }

    std::unique_ptr<FaissCandidate> load_faiss_candidate(const DatabaseState & state) const {
        FILE * file = std::fopen(sidecar_path.c_str(), "rb");
        if (!file) {
            return nullptr;
        }
        try {
            char magic[sizeof(SIDECAR_MAGIC)];
            read_exact(file, magic, sizeof(magic));
            if (std::memcmp(magic, SIDECAR_MAGIC, sizeof(magic)) != 0 ||
                    read_u32(file) != SIDECAR_VERSION ||
                    read_u32(file) != (uint32_t) premise_schema::VERSION) {
                throw std::runtime_error("invalid premise FAISS sidecar header");
            }
            std::array<uint8_t, 16> identity;
            read_exact(file, identity.data(), identity.size());
            const uint64_t next_id = read_u64(file);
            const uint32_t dimensions = read_u32(file);
            const uint64_t row_count = read_u64(file);
            if (identity != state.identity ||
                    next_id == 0 ||
                    next_id > (uint64_t) std::numeric_limits<faiss::idx_t>::max() + 1ULL ||
                    dimensions != (uint32_t) dim ||
                    row_count > (uint64_t) std::numeric_limits<faiss::idx_t>::max()) {
                throw std::runtime_error("premise FAISS sidecar does not match database");
            }

            std::unique_ptr<faiss::Index> serialized(faiss::read_index(file));
            if (std::fgetc(file) != EOF || std::ferror(file)) {
                throw std::runtime_error("invalid trailing premise FAISS sidecar data");
            }
            auto * map = dynamic_cast<faiss::IndexIDMap2 *>(serialized.get());
            auto * flat = map ? dynamic_cast<faiss::IndexFlatIP *>(map->index) : nullptr;
            if (!map || !flat ||
                    map->metric_type != faiss::METRIC_INNER_PRODUCT ||
                    flat->metric_type != faiss::METRIC_INNER_PRODUCT ||
                    map->d != dim || flat->d != dim ||
                    map->ntotal < 0 || (uint64_t) map->ntotal != row_count ||
                    flat->ntotal != map->ntotal ||
                    map->id_map.size() != (size_t) map->ntotal) {
                throw std::runtime_error("invalid premise FAISS sidecar index type");
            }

            auto candidate = std::make_unique<FaissCandidate>();
            candidate->ids.reserve(map->id_map.size());
            for (faiss::idx_t id : map->id_map) {
                if (id <= 0 || (uint64_t) id >= next_id || !candidate->ids.insert(id).second) {
                    throw std::runtime_error("invalid premise FAISS sidecar declaration IDs");
                }
            }
            map->construct_rev_map();
            serialized.release();
            candidate->index.reset(map);
            candidate->identity = state.identity;
            candidate->next_id = next_id;
            candidate->loaded = true;
            const int close_rc = std::fclose(file);
            file = nullptr;
            if (close_rc != 0) {
                throw std::runtime_error("cannot close premise FAISS sidecar");
            }
            return candidate;
        } catch (...) {
            if (file) {
                std::fclose(file);
            }
            return nullptr;
        }
    }

    static void select_active_faiss_ids(
            const DatabaseState & state,
            FaissCandidate & candidate) {
        for (const auto & item : state.rows) {
            if (candidate.ids.find(item.first) == candidate.ids.end()) {
                throw std::runtime_error("premise FAISS sidecar is missing an active declaration ID");
            }
        }

        std::vector<faiss::idx_t> stale;
        for (faiss::idx_t id : candidate.ids) {
            if (state.rows.find(id) == state.rows.end()) {
                stale.push_back(id);
            }
        }
        if (!stale.empty()) {
            faiss::IDSelectorBatch remove(stale.size(), stale.data());
            if (candidate.index->remove_ids(remove) != stale.size()) {
                throw std::runtime_error("failed to remove stale declaration IDs from FAISS");
            }
            for (faiss::idx_t id : stale) {
                candidate.ids.erase(id);
            }
        }
    }

    static void sync_file(FILE * file) {
        if (std::fflush(file) != 0) {
            throw std::runtime_error("cannot flush premise FAISS sidecar");
        }
#if defined(_WIN32)
        if (_commit(_fileno(file)) != 0) {
#else
        if (fsync(fileno(file)) != 0) {
#endif
            throw std::runtime_error("cannot sync premise FAISS sidecar");
        }
    }

    void install_sidecar_file(const std::string & temp_path) const {
#if defined(_WIN32)
        if (!MoveFileExA(temp_path.c_str(), sidecar_path.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            throw std::runtime_error("cannot install premise FAISS sidecar");
        }
#else
        if (std::rename(temp_path.c_str(), sidecar_path.c_str()) != 0) {
            throw std::runtime_error("cannot install premise FAISS sidecar");
        }
        const size_t slash = sidecar_path.find_last_of('/');
        const std::string directory = slash == std::string::npos ? "." :
                (slash == 0 ? "/" : sidecar_path.substr(0, slash));
        int flags = O_RDONLY;
#ifdef O_DIRECTORY
        flags |= O_DIRECTORY;
#endif
        const int directory_fd = open(directory.c_str(), flags);
        if (directory_fd < 0 || fsync(directory_fd) != 0) {
            if (directory_fd >= 0) {
                close(directory_fd);
            }
            throw std::runtime_error("cannot sync premise FAISS sidecar directory");
        }
        close(directory_fd);
#endif
    }

    void persist_faiss_index(
            const faiss::IndexIDMap2 & index,
            const std::unordered_set<faiss::idx_t> & ids,
            const std::array<uint8_t, 16> & identity,
            uint64_t next_id) const {
        if (next_id == 0 ||
                next_id > (uint64_t) std::numeric_limits<faiss::idx_t>::max() + 1ULL ||
                index.ntotal < 0 || (size_t) index.ntotal != ids.size()) {
            throw std::runtime_error("cannot persist invalid premise FAISS candidate");
        }
        for (faiss::idx_t id : ids) {
            if (id <= 0 || (uint64_t) id >= next_id) {
                throw std::runtime_error("cannot persist invalid premise FAISS declaration ID");
            }
        }
        const std::string temp_path = sidecar_path + ".tmp";
        std::remove(temp_path.c_str());
        FILE * file = std::fopen(temp_path.c_str(), "wb");
        if (!file) {
            throw std::runtime_error("cannot open premise FAISS sidecar temporary file");
        }
        try {
            write_exact(file, SIDECAR_MAGIC, sizeof(SIDECAR_MAGIC));
            write_u32(file, SIDECAR_VERSION);
            write_u32(file, (uint32_t) premise_schema::VERSION);
            write_exact(file, identity.data(), identity.size());
            write_u64(file, next_id);
            write_u32(file, (uint32_t) dim);
            write_u64(file, (uint64_t) ids.size());
            faiss::write_index(&index, file);
            sync_file(file);
            const int close_rc = std::fclose(file);
            file = nullptr;
            if (close_rc != 0) {
                throw std::runtime_error("cannot finish premise FAISS sidecar");
            }
            install_sidecar_file(temp_path);
        } catch (...) {
            if (file) {
                std::fclose(file);
            }
            std::remove(temp_path.c_str());
            throw;
        }
    }

    void refresh_database_locked(bool force = false) const {
        const int before = data_version_locked();
        if (!force && before == observed_data_version) {
            return;
        }

        DatabaseState state;
        FaissCandidate candidate;
        int snapshot_data_version = before;
        exec(db, "BEGIN IMMEDIATE");
        try {
            snapshot_data_version = data_version_locked();
            state = read_database_state(db, dim);
            std::unique_ptr<FaissCandidate> loaded = load_faiss_candidate(state);
            if (loaded) {
                candidate = std::move(*loaded);
            } else {
                if (!state.rows.empty()) {
                    throw std::runtime_error("premise database references a missing or invalid FAISS sidecar");
                }
                candidate.index = make_index(dim);
                candidate.identity = state.identity;
                persist_faiss_index(*candidate.index, candidate.ids, candidate.identity, candidate.next_id);
            }
            select_active_faiss_ids(state, candidate);
            exec(db, "COMMIT");
        } catch (...) {
            rollback(db);
            throw;
        }

        rows = std::move(state.rows);
        modules = std::move(state.modules);
        database_identity = state.identity;
        next_faiss_id = candidate.next_id;
        content_revision = state.revision;
        observed_data_version = snapshot_data_version;
        base_index = std::move(candidate.index);
        base_ids = std::move(candidate.ids);
        sidecar_loaded = candidate.loaded;
        scope_cache = {};
    }

    void apply_committed_replacement_locked(
            const std::string & module,
            const std::string & version_token,
            const std::vector<std::string> & imports,
            const std::vector<const Candidate *> & accepted,
            const std::vector<faiss::idx_t> & new_ids) {
        std::vector<faiss::idx_t> old_ids;
        auto existing = modules.find(module);
        if (existing != modules.end()) {
            old_ids = existing->second.declaration_ids;
        }

        std::vector<faiss::idx_t> old_base_ids;
        old_base_ids.reserve(old_ids.size());
        for (faiss::idx_t id : old_ids) {
            if (base_ids.find(id) != base_ids.end()) {
                old_base_ids.push_back(id);
            }
        }
        if (!old_base_ids.empty()) {
            faiss::IDSelectorBatch remove(old_base_ids.size(), old_base_ids.data());
            if (base_index->remove_ids(remove) != old_base_ids.size()) {
                throw std::runtime_error("failed to remove stale declaration IDs from FAISS");
            }
        }

        for (faiss::idx_t id : old_ids) {
            rows.erase(id);
            base_ids.erase(id);
        }

        ModuleEntry & entry = modules[module];
        entry.version_token = version_token;
        entry.imports = imports;
        entry.declaration_ids = new_ids;
        for (size_t i = 0; i < new_ids.size(); ++i) {
            const faiss::idx_t id = new_ids[i];
            rows.emplace(id, Row{accepted[i]->name, module});
        }
        ++content_revision;
        scope_cache = {};
    }

    std::vector<Hit> search_base(
            const float * query,
            int top_k,
            const std::vector<faiss::idx_t> * eligible_ids) const {
        if (!base_index || base_ids.empty() || top_k <= 0) {
            return {};
        }

        size_t available = base_ids.size();
        if (eligible_ids) {
            available = 0;
            for (faiss::idx_t id : *eligible_ids) {
                available += base_ids.find(id) != base_ids.end();
            }
        }
        const faiss::idx_t count = (faiss::idx_t) std::min<size_t>((size_t) top_k, available);
        if (count <= 0) {
            return {};
        }

        std::vector<float> scores((size_t) count);
        std::vector<faiss::idx_t> labels((size_t) count);
        if (eligible_ids) {
            faiss::IDSelectorBatch allow(eligible_ids->size(), eligible_ids->data());
            faiss::SearchParameters parameters;
            parameters.sel = &allow;
            base_index->search(1, query, count, scores.data(), labels.data(), &parameters);
        } else {
            base_index->search(1, query, count, scores.data(), labels.data());
        }

        std::vector<Hit> hits;
        hits.reserve((size_t) count);
        for (faiss::idx_t i = 0; i < count; ++i) {
            const faiss::idx_t id = labels[(size_t) i];
            auto row = rows.find(id);
            if (base_ids.find(id) != base_ids.end() && row != rows.end()) {
                hits.push_back({row->second.name, row->second.module, scores[(size_t) i]});
            }
        }
        return hits;
    }

    void score_locals(
            const float * query,
            const std::vector<Candidate> & locals,
            const std::unordered_set<std::string> * indexed_names,
            std::vector<Hit> & hits) const {
        std::unordered_set<std::string> local_names;
        for (const auto & local : locals) {
            if (local.name.empty() ||
                    (indexed_names && indexed_names->find(local.name) != indexed_names->end()) ||
                    !local_names.insert(local.name).second) {
                continue;
            }
            validate_vector(local.embedding, dim, "local embedding");
            float score = 0.0f;
            for (int i = 0; i < dim; ++i) {
                score += query[i] * local.embedding[(size_t) i];
            }
            hits.push_back({local.name, local.module, score});
        }
    }

    static void trim_hits(std::vector<Hit> & hits, int top_k) {
        const size_t count = std::min((size_t) top_k, hits.size());
        if (count < hits.size()) {
            std::partial_sort(hits.begin(), hits.begin() + count, hits.end(),
                    [](const Hit & a, const Hit & b) { return a.score > b.score; });
            hits.resize(count);
        } else {
            std::sort(hits.begin(), hits.end(),
                    [](const Hit & a, const Hit & b) { return a.score > b.score; });
        }
    }

    void collect_candidate_rows(
            const std::string & module,
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
        for (const auto & imported : it->second.imports) {
            collect_candidate_rows(imported, visited, taken_names, candidate_rows);
        }
        for (faiss::idx_t id : it->second.declaration_ids) {
            auto row = rows.find(id);
            if (row != rows.end() && !row->second.name.empty() &&
                    taken_names.insert(row->second.name).second) {
                candidate_rows.push_back(id);
            }
        }
    }
};
