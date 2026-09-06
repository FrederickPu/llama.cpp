#include "premise-index.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static PremiseIndex::Candidate candidate(
        const std::string & name, float x, float y, const std::string & module = {}) {
    return {name, module, {x, y}};
}

static bool has_hit(const std::vector<PremiseIndex::Hit> & hits, const std::string & name) {
    for (const auto & hit : hits) {
        if (hit.name == name) {
            return true;
        }
    }
    return false;
}

static void remove_database(const std::filesystem::path & path) {
    std::remove(path.string().c_str());
    std::remove((path.string() + "-shm").c_str());
    std::remove((path.string() + "-wal").c_str());
    std::remove((path.string() + ".faiss").c_str());
    std::remove((path.string() + ".faiss.tmp").c_str());
}

static void sqlite_exec_checked(sqlite3 * db, const char * sql) {
    char * error = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &error) != SQLITE_OK) {
        const std::string message = error ? error : sqlite3_errmsg(db);
        sqlite3_free(error);
        throw std::runtime_error(message);
    }
}

int main() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto temp = std::filesystem::temp_directory_path();
    const auto path = temp / ("llama-premise-index-test-" + std::to_string(suffix) + ".db");
    const auto worker_path = temp / ("llama-premise-index-worker-" + std::to_string(suffix) + ".db");
    const auto stale_path = temp / ("llama-premise-index-stale-" + std::to_string(suffix) + ".db");
    const auto corrupt_path = temp / ("llama-premise-index-corrupt-" + std::to_string(suffix) + ".db");
    const auto identity_a_path = temp / ("llama-premise-index-identity-a-" + std::to_string(suffix) + ".db");
    const auto identity_b_path = temp / ("llama-premise-index-identity-b-" + std::to_string(suffix) + ".db");
    const auto rollback_path = temp / ("llama-premise-index-rollback-" + std::to_string(suffix) + ".db");
    const auto rollback_snapshot_path = temp / ("llama-premise-index-rollback-snapshot-" + std::to_string(suffix) + ".db");
    const auto v1_path = temp / ("llama-premise-index-v1-" + std::to_string(suffix) + ".db");
    remove_database(path);
    remove_database(worker_path);
    remove_database(stale_path);
    remove_database(corrupt_path);
    remove_database(identity_a_path);
    remove_database(identity_b_path);
    remove_database(rollback_path);
    remove_database(rollback_snapshot_path);
    remove_database(v1_path);

    try {
        const float query_x[] = {1.0f, 0.0f};
        const float query_y[] = {0.0f, 1.0f};

        {
            PremiseIndex index;
            index.open(path.string(), 2);
            require(std::filesystem::exists(path.string() + ".faiss"),
                    "first open did not persist a FAISS sidecar");
            require(!index.loaded_sidecar_for_test(), "first open unexpectedly loaded a sidecar");
            std::vector<float> local_embedding;
            require(!index.find_local_embedding("local declaration", local_embedding),
                    "unknown local embedding was cached");
            index.cache_local_embedding("local declaration", {0.6f, 0.8f});
            require(index.find_local_embedding("local declaration", local_embedding) &&
                    local_embedding == std::vector<float>({0.6f, 0.8f}),
                    "local embedding cache lookup failed");
            index.cache_local_embedding("local declaration", {1.0f, 0.0f});
            require(index.find_local_embedding("local declaration", local_embedding) &&
                    local_embedding == std::vector<float>({1.0f, 0.0f}),
                    "local embedding cache update failed");
            index.replace_module("A", "a1", {}, {
                candidate("A.x", 1.0f, 0.0f),
                candidate("A.y", 0.0f, 1.0f),
            });
            index.replace_module("B", "b1", {"A"}, {
                candidate("A.x", 0.0f, 1.0f),
                candidate("B.z", -1.0f, 0.0f),
            });

            require(index.size() == 4, "unexpected initial row count");
            require(index.get_module_version("A") == "a1", "module version was not stored");

            auto global = index.search_global(query_x, 1);
            require(global.size() == 1 && global[0].name == "A.x", "FAISS KNN result mismatch");
            require(std::fabs(global[0].score - 1.0f) < 1e-5f, "FAISS KNN score mismatch");

            auto scoped = index.search_scoped(query_y, {"B"}, {}, 8);
            require(scoped.size() == 3, "scoped search did not apply name shadowing");
            require(scoped[0].name == "A.y", "scoped KNN ranking mismatch");

            index.replace_module("A", "a2", {}, {
                candidate("A.new", -1.0f, 0.0f),
            });
            require(index.size() == 3, "module replacement left stale rows");
            require(index.get_module_version("B") == "b1", "replacement changed another module");
            global = index.search_global(query_x, 8);
            require(!has_hit(global, "A.y"), "deleted declaration remained searchable");

            scoped = index.search_scoped(query_y, {"B"}, {}, 8);
            require(scoped.size() == 3, "replacement produced an invalid scoped result set");
            require(scoped[0].name == "A.x", "replacement did not update shadowing");

            try {
                index.replace_module("A", "broken", {}, {
                    candidate("bad", std::numeric_limits<float>::quiet_NaN(), 0.0f),
                });
                require(false, "invalid vector replacement unexpectedly succeeded");
            } catch (const std::exception &) {
                require(index.get_module_version("A") == "a2", "failed replacement was not rolled back");
                require(index.size() == 3, "failed replacement changed row count");
            }
        }

        {
            PremiseIndex index;
            index.open(path.string(), 2);
            require(index.loaded_sidecar_for_test(), "second open did not load the FAISS sidecar");
            require(index.size() == 3, "rows did not survive database reopen");
            require(index.get_module_version("A") == "a2", "version did not survive database reopen");
            std::vector<float> local_embedding;
            require(!index.find_local_embedding("local declaration", local_embedding),
                    "local embedding cache was serialized");
            auto global = index.search_global(query_y, 3);
            require(global.size() == 3 && has_hit(global, "A.x"), "sidecar startup results changed");

            index.replace_module("Delta", "d1", {}, {
                candidate("Delta.diagonal", 0.6f, 0.8f),
            });
            global = index.search_global(query_y, 2);
            require(has_hit(global, "A.x") && has_hit(global, "Delta.diagonal"),
                    "newly flushed FAISS results were not searchable");

            const std::string nul_module("N\0M", 3);
            const std::string nul_token("t\0x", 3);
            {
                PremiseIndex writer;
                writer.open(path.string(), 2);
                writer.replace_module(nul_module, nul_token, {}, {
                    candidate(std::string("n\0d", 3), 0.0f, 1.0f),
                });
            }
            require(index.get_module_version(nul_module) == nul_token,
                    "metadata did not refresh after an external commit");
            auto external = index.search_scoped(query_y, {nul_module}, {}, 1);
            require(external.size() == 1 && external[0].name == std::string("n\0d", 3),
                    "embedded NUL metadata was not preserved");

            const std::string injection = "X'); DROP TABLE premise_modules; --";
            index.replace_module(injection, injection, {injection}, {
                candidate(injection, 1.0f, 0.0f),
            });
            require(index.get_module_version(injection) == injection,
                    "SQL-like metadata was not stored literally");
            require(index.get_module_version("A") == "a2",
                    "SQL-like metadata changed the premise schema");

            std::vector<PremiseIndex::Candidate> bulk;
            bulk.reserve(4100);
            for (int i = 0; i < 4100; ++i) {
                bulk.push_back(candidate("Bulk." + std::to_string(i), 1.0f, 0.0f));
            }
            index.replace_module("Bulk", "bulk1", {}, bulk);
            auto large = index.search_scoped(query_x, {"Bulk"}, {}, 4097);
            require(large.size() == 4097, "large scoped KNN query was truncated");
        }

        {
            PremiseIndex index;
            index.open(path.string(), 2);
            auto large = index.search_scoped(query_x, {"Bulk"}, {}, 4097);
            require(large.size() == 4097, "large base scoped KNN query was truncated");
            large = index.search_global(query_x, 4097);
            require(large.size() == 4097, "large base global KNN query was truncated");
        }

        {
            sqlite3 * raw = nullptr;
            require(sqlite3_open_v2(path.string().c_str(), &raw, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK,
                    "could not inspect premise database");
            sqlite3_stmt * statement = nullptr;
            require(sqlite3_prepare_v2(raw,
                    "SELECT 1 FROM sqlite_schema WHERE name = 'premise_declaration_embeddings'",
                    -1, &statement, nullptr) == SQLITE_OK,
                    "could not inspect embedding schema");
            require(sqlite3_step(statement) == SQLITE_DONE,
                    "declaration embeddings were duplicated in SQLite");
            sqlite3_finalize(statement);
            require(sqlite3_prepare_v2(raw,
                    "SELECT schema_version, length(database_identity) FROM premise_config WHERE id = 1",
                    -1, &statement, nullptr) == SQLITE_OK,
                    "could not inspect premise configuration");
            require(sqlite3_step(statement) == SQLITE_ROW &&
                    sqlite3_column_int(statement, 0) == 4 && sqlite3_column_int(statement, 1) == 16,
                    "premise database identity or schema version is invalid");
            sqlite3_finalize(statement);
            sqlite3_close_v2(raw);
        }

        {
            PremiseIndex index;
            index.open(stale_path.string(), 2);
            index.replace_module("Stale", "v1", {}, {
                candidate("Stale.old", 1.0f, 0.0f),
            });
        }
        {
            PremiseIndex index;
            index.open(stale_path.string(), 2);
            require(index.loaded_sidecar_for_test(), "current sidecar was not loaded before replacement");
            index.replace_module("Stale", "v2", {}, {
                candidate("Stale.new", 0.0f, 1.0f),
            });
        }
        {
            PremiseIndex index;
            index.open(stale_path.string(), 2);
            require(index.loaded_sidecar_for_test(), "stale sidecar was not loaded");
            const auto hits = index.search_global(query_y, 8);
            require(hits.size() == 1 && has_hit(hits, "Stale.new") && !has_hit(hits, "Stale.old"),
                    "inactive sidecar IDs returned replaced declarations");
        }

        {
            PremiseIndex index;
            index.open(corrupt_path.string(), 2);
            index.replace_module("Corrupt", "v1", {}, {
                candidate("Corrupt.row", 1.0f, 0.0f),
            });
        }
        {
            std::ofstream corrupt(corrupt_path.string() + ".faiss", std::ios::binary | std::ios::trunc);
            corrupt << "not a FAISS sidecar";
            require((bool) corrupt, "could not corrupt FAISS sidecar");
        }
        {
            bool rejected = false;
            try {
                PremiseIndex index;
                index.open(corrupt_path.string(), 2);
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected, "corrupt sidecar was accepted without its FAISS vectors");
        }

        {
            PremiseIndex index;
            index.open(identity_a_path.string(), 2);
            index.replace_module("IdentityA", "v1", {}, {
                candidate("IdentityA.row", 1.0f, 0.0f),
            });
        }
        {
            PremiseIndex index;
            index.open(identity_b_path.string(), 2);
            index.replace_module("IdentityB", "v1", {}, {
                candidate("IdentityB.row", 0.0f, 1.0f),
            });
        }
        std::filesystem::copy_file(identity_a_path.string() + ".faiss",
                identity_b_path.string() + ".faiss", std::filesystem::copy_options::overwrite_existing);
        {
            bool rejected = false;
            try {
                PremiseIndex index;
                index.open(identity_b_path.string(), 2);
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected, "sidecar from another database was accepted");
        }

        {
            PremiseIndex index;
            index.open(rollback_path.string(), 2);
            index.replace_module("Rollback", "v1", {}, {
                candidate("Rollback.old", 1.0f, 0.0f),
            });
        }
        std::filesystem::copy_file(rollback_path, rollback_snapshot_path,
                std::filesystem::copy_options::overwrite_existing);
        {
            PremiseIndex index;
            index.open(rollback_path.string(), 2);
            index.replace_module("Rollback", "v2", {}, {
                candidate("Rollback.new", 0.0f, 1.0f),
            });
        }
        std::remove((rollback_path.string() + "-shm").c_str());
        std::remove((rollback_path.string() + "-wal").c_str());
        std::filesystem::copy_file(rollback_snapshot_path, rollback_path,
                std::filesystem::copy_options::overwrite_existing);
        {
            PremiseIndex index;
            index.open(rollback_path.string(), 2);
            require(index.get_module_version("Rollback") == "v1",
                    "pre-commit SQLite state did not remain active");
            auto hits = index.search_global(query_x, 2);
            require(hits.size() == 1 && hits[0].name == "Rollback.old",
                    "inactive preflushed FAISS ID changed the committed state");
            index.replace_module("Rollback", "v2-retry", {}, {
                candidate("Rollback.retry", 0.0f, 1.0f),
            });
            hits = index.search_global(query_y, 2);
            require(hits.size() == 1 && hits[0].name == "Rollback.retry",
                    "FAISS ID allocator reused an orphaned ID");
        }
        {
            sqlite3 * raw = nullptr;
            require(sqlite3_open_v2(rollback_path.string().c_str(), &raw,
                    SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK,
                    "could not inspect rollback database");
            sqlite3_stmt * statement = nullptr;
            require(sqlite3_prepare_v2(raw,
                    "SELECT id FROM premise_declarations WHERE module = 'Rollback'",
                    -1, &statement, nullptr) == SQLITE_OK,
                    "could not inspect retry FAISS ID");
            require(sqlite3_step(statement) == SQLITE_ROW && sqlite3_column_int64(statement, 0) == 3,
                    "FAISS ID allocator reused an orphaned ID");
            sqlite3_finalize(statement);
            sqlite3_close_v2(raw);
        }

        {
            PremiseIndex index;
            index.open(worker_path.string(), 2);
            index.replace_module("Base", "v1", {}, {
                candidate("Base.x", 1.0f, 0.0f),
                candidate("Base.y", 0.0f, 1.0f),
            });
            auto hits = index.search_global(query_x, 2);
            require(hits.size() == 2 && has_hit(hits, "Base.x"),
                    "FAISS-first write lost declarations");

            index.replace_module("Base", "v2", {}, {
                candidate("Base.new", 0.0f, 1.0f),
            });
            index.replace_module("Tail", "v1", {}, {
                candidate("Tail.row", 1.0f, 0.0f),
            });
            hits = index.search_global(query_x, 4);
            require(hits.size() == 2 && has_hit(hits, "Base.new") && has_hit(hits, "Tail.row"),
                    "replacement did not remove inactive IDs immediately");
        }

        {
            PremiseIndex index;
            index.open(worker_path.string(), 2);
            require(index.loaded_sidecar_for_test(), "replacement did not update the sidecar");
            const auto hits = index.search_global(query_x, 4);
            require(hits.size() == 2 && has_hit(hits, "Base.new") && has_hit(hits, "Tail.row"),
                    "durable sidecar did not preserve current rows");
        }

        {
            PremiseIndex index;
            index.open(worker_path.string(), 2);
            std::vector<PremiseIndex::Candidate> pending;
            pending.reserve(2000);
            for (int i = 0; i < 2000; ++i) {
                pending.push_back(candidate("Pending." + std::to_string(i), 1.0f, 0.0f));
            }
            index.replace_module("Pending", "v1", {}, pending);
            index.replace_module("Pending", "v2", {}, {
                candidate("Pending.current", 0.0f, 1.0f),
            });
        }

        {
            sqlite3 * raw = nullptr;
            require(sqlite3_open_v2(v1_path.string().c_str(), &raw,
                    SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) == SQLITE_OK,
                    "could not create schema v1 test database");
            sqlite_exec_checked(raw,
                    "CREATE TABLE premise_config("
                    "id INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL, embedding_dim INTEGER NOT NULL);"
                    "INSERT INTO premise_config VALUES(1, 1, 2);");
            sqlite3_close_v2(raw);

            bool rejected = false;
            try {
                PremiseIndex incompatible;
                incompatible.open(v1_path.string(), 2);
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected, "schema v1 database was not rejected");
        }
    } catch (const std::exception & error) {
        fprintf(stderr, "test-premise-index: %s\n", error.what());
        remove_database(path);
        remove_database(worker_path);
        remove_database(stale_path);
        remove_database(corrupt_path);
        remove_database(identity_a_path);
        remove_database(identity_b_path);
        remove_database(rollback_path);
        remove_database(rollback_snapshot_path);
        remove_database(v1_path);
        return 1;
    }

    remove_database(path);
    remove_database(worker_path);
    remove_database(stale_path);
    remove_database(corrupt_path);
    remove_database(identity_a_path);
    remove_database(identity_b_path);
    remove_database(rollback_path);
    remove_database(rollback_snapshot_path);
    remove_database(v1_path);
    return 0;
}
