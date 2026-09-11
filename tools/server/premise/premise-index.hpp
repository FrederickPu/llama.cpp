#pragma once

#include "premise-schema.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/IDSelector.h>
#include <sqlite3.h>

#include <algorithm>
#include <cctype>
#include <cmath>
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

    std::vector<std::string> get_module_versions(const std::vector<std::string> & requested) const {
        std::lock_guard<std::mutex> lock(mu);
        require_db();
        refresh_database_locked();
        std::vector<std::string> versions;
        versions.reserve(requested.size());
        for (const auto & module : requested) {
            auto it = modules.find(module);
            versions.push_back(it == modules.end() ? std::string() : it->second.version_token);
        }
        return versions;
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
                        "INSERT INTO premise_declarations(module, ordinal, name) VALUES(?1, ?2, ?3)");
                Statement insert_embedding(db,
                        "INSERT INTO premise_declaration_embeddings(declaration_id, embedding) VALUES(?1, ?2)");
                new_ids.clear();
                new_ids.reserve(accepted.size());
                for (size_t i = 0; i < accepted.size(); ++i) {
                    insert_declaration.bind_text(1, module);
                    insert_declaration.bind_int64(2, checked_int64(i, "too many declarations"));
                    insert_declaration.bind_text(3, accepted[i]->name);
                    insert_declaration.run();
                    insert_declaration.reset();

                    const faiss::idx_t id = sqlite3_last_insert_rowid(db);
                    if (id <= 0) {
                        throw std::runtime_error("invalid premise declaration ID");
                    }
                    insert_embedding.bind_int64(1, id);
                    insert_embedding.bind_vector(2, accepted[i]->embedding);
                    insert_embedding.run();
                    insert_embedding.reset();
                    new_ids.push_back(id);
                }

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
        int64_t revision = 0;
    };

    struct BuiltIndex {
        std::unique_ptr<faiss::IndexIDMap2> index;
        std::unordered_set<faiss::idx_t> ids;
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

        void bind_vector(int index, const std::vector<float> & value) {
            const size_t bytes = value.size() * sizeof(float);
            if (bytes > (size_t) std::numeric_limits<int>::max()) {
                throw std::runtime_error("embedding is too large for SQLite");
            }
            check(sqlite3_bind_blob(stmt, index, value.data(), (int) bytes, SQLITE_TRANSIENT));
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
                        "SELECT schema_version, embedding_dim "
                        "FROM premise_config WHERE id = 1");
                if (!read_config.step()) {
                    throw std::runtime_error("premise database has no configuration row");
                }
                const int stored_schema = sqlite3_column_int(read_config.stmt, 0);
                const int stored_dim = sqlite3_column_int(read_config.stmt, 1);
                if (read_config.step()) {
                    throw std::runtime_error("premise database has multiple configuration rows");
                }
                if (stored_schema != premise_schema::VERSION) {
                    throw std::runtime_error("unsupported premise database schema version");
                }
                if (stored_dim != dim) {
                    throw std::runtime_error("premise database embedding dimension does not match model");
                }
                exec(db, premise_schema::SQL);
            } else {
                exec(db, premise_schema::SQL);
                Statement insert_config(db,
                        "INSERT INTO premise_config("
                        "id, schema_version, embedding_dim, content_revision) "
                        "VALUES(1, ?1, ?2, 0)");
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

    static DatabaseState read_database_state(sqlite3 * database, int dimensions) {
        DatabaseState state;
        {
            Statement config(database,
                    "SELECT schema_version, embedding_dim, content_revision "
                    "FROM premise_config WHERE id = 1");
            if (!config.step() ||
                    sqlite3_column_type(config.stmt, 0) != SQLITE_INTEGER ||
                    sqlite3_column_type(config.stmt, 1) != SQLITE_INTEGER ||
                    sqlite3_column_type(config.stmt, 2) != SQLITE_INTEGER ||
                    sqlite3_column_int(config.stmt, 0) != premise_schema::VERSION ||
                    sqlite3_column_int(config.stmt, 1) != dimensions) {
                throw std::runtime_error("invalid premise database configuration");
            }
            state.revision = sqlite3_column_int64(config.stmt, 2);
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

    static void validate_stored_vector(const void * blob, int bytes, int dimensions) {
        if (!blob || bytes != dimensions * (int) sizeof(float)) {
            throw std::runtime_error("stored embedding dimension mismatch");
        }
        const uint8_t * data = static_cast<const uint8_t *>(blob);
        double norm_squared = 0.0;
        for (int i = 0; i < dimensions; ++i) {
            float value;
            std::memcpy(&value, data + (size_t) i * sizeof(float), sizeof(value));
            if (!std::isfinite(value)) {
                throw std::runtime_error("stored embedding contains a non-finite component");
            }
            norm_squared += (double) value * value;
        }
        if (!std::isfinite(norm_squared) || std::fabs(norm_squared - 1.0) > 1e-3) {
            throw std::runtime_error("stored embedding is not approximately unit normalized");
        }
    }

    static BuiltIndex build_index(
            sqlite3 * database, const DatabaseState & state, int dimensions) {
        BuiltIndex built;
        built.index = make_index(dimensions);
        built.ids.reserve(state.rows.size());

        constexpr size_t batch_size = 1024;
        std::vector<faiss::idx_t> ids;
        std::vector<float> embeddings;
        ids.reserve(std::min(batch_size, state.rows.size()));
        if ((size_t) dimensions > std::numeric_limits<size_t>::max() / batch_size) {
            throw std::runtime_error("premise embedding dimension is too large");
        }
        embeddings.reserve(std::min(batch_size, state.rows.size()) * (size_t) dimensions);

        Statement read_embeddings(database,
                "SELECT declaration_id, embedding "
                "FROM premise_declaration_embeddings ORDER BY declaration_id");
        while (read_embeddings.step()) {
            const faiss::idx_t id = sqlite3_column_int64(read_embeddings.stmt, 0);
            const int bytes = sqlite3_column_bytes(read_embeddings.stmt, 1);
            const void * blob = sqlite3_column_blob(read_embeddings.stmt, 1);
            if (state.rows.find(id) == state.rows.end() || !built.ids.insert(id).second) {
                throw std::runtime_error("invalid declaration embedding ID");
            }
            validate_stored_vector(blob, bytes, dimensions);
            ids.push_back(id);
            const size_t offset = embeddings.size();
            embeddings.resize(offset + (size_t) dimensions);
            std::memcpy(embeddings.data() + offset, blob, (size_t) bytes);
            if (ids.size() == batch_size) {
                built.index->add_with_ids((faiss::idx_t) ids.size(), embeddings.data(), ids.data());
                ids.clear();
                embeddings.clear();
            }
        }
        if (!ids.empty()) {
            built.index->add_with_ids((faiss::idx_t) ids.size(), embeddings.data(), ids.data());
        }
        if (built.ids.size() != state.rows.size()) {
            throw std::runtime_error("premise embedding/declaration row count mismatch");
        }
        return built;
    }

    void refresh_database_locked(bool force = false) const {
        int before = data_version_locked();
        if (!force && before == observed_data_version) {
            return;
        }

        for (;;) {
            DatabaseState state;
            BuiltIndex built;
            exec(db, "BEGIN");
            try {
                state = read_database_state(db, dim);
                built = build_index(db, state, dim);
                exec(db, "COMMIT");
            } catch (...) {
                rollback(db);
                throw;
            }
            const int after = data_version_locked();
            if (before != after) {
                before = after;
                continue;
            }

            rows = std::move(state.rows);
            modules = std::move(state.modules);
            content_revision = state.revision;
            observed_data_version = after;
            base_index = std::move(built.index);
            base_ids = std::move(built.ids);
            scope_cache = {};
            return;
        }
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
        if (!new_ids.empty()) {
            std::vector<float> embeddings;
            embeddings.reserve(new_ids.size() * (size_t) dim);
            for (const Candidate * candidate : accepted) {
                embeddings.insert(embeddings.end(),
                        candidate->embedding.begin(), candidate->embedding.end());
            }
            base_index->add_with_ids(
                    (faiss::idx_t) new_ids.size(), embeddings.data(), new_ids.data());
            base_ids.insert(new_ids.begin(), new_ids.end());
        }
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
