include(FetchContent)

license_add_file("SQLite" "SQLITE-PUBLIC-DOMAIN.txt")

FetchContent_Declare(sqlite_amalgamation
    URL https://www.sqlite.org/2026/sqlite-amalgamation-3530100.zip
    URL_HASH SHA256=36ad6e7f38540a3b21a2ac36340833f0a9e426bc1c752751c3ba669466827eae
)
FetchContent_MakeAvailable(sqlite_amalgamation)

set(_sqlite_source_dir "${sqlite_amalgamation_SOURCE_DIR}")
if (EXISTS "${_sqlite_source_dir}/sqlite-amalgamation-3530100/sqlite3.c")
    set(_sqlite_source_dir "${_sqlite_source_dir}/sqlite-amalgamation-3530100")
endif()

add_library(llama-sqlite3 STATIC "${_sqlite_source_dir}/sqlite3.c")
target_include_directories(llama-sqlite3 PUBLIC "${_sqlite_source_dir}")
target_compile_definitions(llama-sqlite3 PRIVATE
    SQLITE_OMIT_LOAD_EXTENSION
    SQLITE_THREADSAFE=1
)

if (MSVC)
    target_compile_options(llama-sqlite3 PRIVATE /w)
else()
    target_compile_options(llama-sqlite3 PRIVATE -w)
endif()

if (UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(llama-sqlite3 PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()
