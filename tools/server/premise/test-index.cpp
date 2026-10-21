#include "premise-index.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using decl = premise_index::decl;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::vector<std::string> names(const std::vector<premise_index::hit> & hits) {
    std::vector<std::string> out;
    for (const auto & hit : hits) {
        out.push_back(hit.name);
    }
    return out;
}

static bool rejects(const std::function<void()> & action) {
    try {
        action();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

static void exec(const std::filesystem::path & path, const char * sql) {
    sqlite3 * db = nullptr;
    require(sqlite3_open(path.string().c_str(), &db) == SQLITE_OK, "could not open test database");
    const int rc = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
    sqlite3_close(db);
    require(rc == SQLITE_OK, "could not modify test database");
}

static int query_int(const std::filesystem::path & path, const char * sql) {
    sqlite3 * db = nullptr;
    sqlite3_stmt * stmt = nullptr;
    require(sqlite3_open(path.string().c_str(), &db) == SQLITE_OK, "could not open test database");
    require(sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW,
            "could not query test database");
    const int value = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return value;
}

static void run(const std::filesystem::path & dir) {
    const auto path = dir / "premise.db";
    const std::vector<float> x = {1.0f, 0.0f};
    const std::vector<float> y = {0.0f, 1.0f};

    {
        premise_index index(path.string(), 2);
        index.replace("A", "a1", {{"A.x", x}, {"A.y", y}, {"A.y", x}, {"", y}});
        index.replace("B", "b1", {{"B.x", x}, {"shared", y}});
        index.replace("C", "c1", {{"shared", y}, {"C.y", {0.6f, 0.8f}}});
        require(index.version("A") == "a1" && index.version("Missing").empty(), "module versions");

        auto hits = index.search(y, {"A", "B", "C"}, {}, 10);
        require(names(hits) == std::vector<std::string>({"A.y", "shared", "C.y", "A.x", "B.x"}),
                "repeated and empty names must be dropped, ranking by score then name");
        require(std::fabs(hits[0].score - 1.0f) < 1e-6f && std::fabs(hits[2].score - 0.8f) < 1e-6f,
                "scores are dot products");
        require(names(index.search(y, {"A", "B", "C"}, {}, 2)) == std::vector<std::string>({"A.y", "shared"}),
                "search returns the k best");
        require(names(index.search(y, {"B"}, {}, 10)) == std::vector<std::string>({"shared", "B.x"}),
                "only the selected modules are searched");
        require(names(index.search(y, {"Missing"}, {{"local", {0.8f, 0.6f}}, {"B.x", y}}, 10)) ==
                std::vector<std::string>({"B.x", "local"}),
                "local declarations are searched too");
        require(names(index.search(y, {"B"}, {{"B.x", y}}, 10)) == std::vector<std::string>({"B.x", "shared"}),
                "a local declaration and a cached one with the same name give one hit, the better one");
        require(index.search(y, {"A"}, {}, 0).empty(), "k = 0 gives no hits");

        index.replace("A", "a2", {{"A.new", y}});
        require(names(index.search(y, {"A"}, {}, 10)) == std::vector<std::string>({"A.new"}),
                "replacing a module drops its old declarations");
        require(index.version("B") == "b1", "replacing a module leaves the others");

        require(rejects([&] { index.replace("A", "bad", {{"nan", {std::numeric_limits<float>::quiet_NaN(), 0.0f}}}); }) &&
                rejects([&] { index.replace("A", "bad", {{"long", {1.0f, 1.0f}}}); }) &&
                rejects([&] { index.replace("A", "bad", {{"short", {1.0f}}}); }),
                "invalid embeddings are rejected");
        require(index.version("A") == "a2" && names(index.search(y, {"A"}, {}, 10)) == std::vector<std::string>({"A.new"}),
                "a rejected replacement changes nothing");
        require(rejects([&] { index.search({2.0f, 0.0f}, {"A"}, {}, 1); }), "invalid queries are rejected");

        const std::string nul_module("N\0M", 3);
        const std::string nul_name("n\0d", 3);
        index.replace(nul_module, std::string("t\0x", 3), {{nul_name, y}});
        index.replace("X'); DROP TABLE premise_modules; --", "v", {{"inject", y}});
    }

    {
        premise_index index(path.string(), 2);
        require(index.version("A") == "a2" && index.version("C") == "c1", "versions persist");
        const auto hits = index.search(y, {"A", "B", "C"}, {}, 10);
        require(names(hits) == std::vector<std::string>({"A.new", "shared", "C.y", "B.x"}), "declarations persist");
        require(std::fabs(hits[2].score - 0.8f) < 1e-3f, "embeddings persist to f16 precision");
        require(index.version(std::string("N\0M", 3)) == std::string("t\0x", 3) &&
                names(index.search(y, {std::string("N\0M", 3)}, {}, 1)) == std::vector<std::string>({std::string("n\0d", 3)}),
                "text with NUL bytes persists");
        require(index.version("X'); DROP TABLE premise_modules; --") == "v", "text that looks like SQL persists");
    }
    require(query_int(path, "SELECT length(embedding) FROM premise_declaration_embeddings LIMIT 1") == 2 * 2,
            "embeddings are stored as f16");
    require(query_int(path, "SELECT schema_version FROM premise_config") == premise_schema::VERSION,
            "the schema version is recorded");

    require(rejects([&] { premise_index index(path.string(), 3); }), "a different embedding dimension is rejected");

    exec(path, "UPDATE premise_declaration_embeddings SET embedding = zeroblob(3)");
    require(rejects([&] { premise_index index(path.string(), 2); }), "corrupt embeddings are rejected");

    const auto old = dir / "old.db";
    exec(old, "CREATE TABLE premise_config(id INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL, "
              "embedding_dim INTEGER NOT NULL); INSERT INTO premise_config VALUES(1, 1, 2);");
    require(rejects([&] { premise_index index(old.string(), 2); }), "other schema versions are rejected");
    require(query_int(old, "SELECT count(*) FROM sqlite_schema") == 1, "a rejected database is left unmodified");
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("llama-premise-index-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    int status = 0;
    try {
        run(dir);
    } catch (const std::exception & e) {
        fprintf(stderr, "test-premise-index: %s\n", e.what());
        status = 1;
    }
    std::filesystem::remove_all(dir);
    return status;
}
