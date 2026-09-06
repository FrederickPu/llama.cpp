#pragma once

namespace premise_schema {

inline constexpr int VERSION = 4;

inline constexpr const char * SQL = R"sql(
CREATE TABLE IF NOT EXISTS premise_config(
    id INTEGER PRIMARY KEY CHECK(id = 1),
    schema_version INTEGER NOT NULL,
    embedding_dim INTEGER NOT NULL,
    content_revision INTEGER NOT NULL DEFAULT 0,
    database_identity BLOB NOT NULL CHECK(length(database_identity) = 16)
);
CREATE TABLE IF NOT EXISTS premise_modules(
    module TEXT PRIMARY KEY,
    version_token TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS premise_imports(
    module TEXT NOT NULL REFERENCES premise_modules(module) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    imported_module TEXT NOT NULL,
    PRIMARY KEY(module, ordinal)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS premise_declarations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    module TEXT NOT NULL REFERENCES premise_modules(module) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    name TEXT NOT NULL,
    UNIQUE(module, ordinal),
    UNIQUE(module, name)
);
CREATE INDEX IF NOT EXISTS premise_declarations_module
    ON premise_declarations(module, ordinal);
)sql";

} // namespace premise_schema
