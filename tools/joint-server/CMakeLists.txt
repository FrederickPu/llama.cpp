# tools/joint-server/CMakeLists.txt
#
# Joint retrieval + generation server: wraps llama.cpp server infrastructure
# with premise selection hooks and Lean protocol endpoints.

set(_SERVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../server")
set(_ROOT_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_executable(joint-server
    ${CMAKE_CURRENT_SOURCE_DIR}/server.cpp
    ${_SERVER_DIR}/server-context.cpp
    ${_SERVER_DIR}/server-http.cpp
    ${_SERVER_DIR}/server-chat.cpp
    ${_SERVER_DIR}/server-task.cpp
    ${_SERVER_DIR}/server-queue.cpp
    ${_SERVER_DIR}/server-common.cpp
    ${_SERVER_DIR}/server-tools.cpp
    ${_SERVER_DIR}/server-models.cpp
)

target_include_directories(joint-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../premise-server
    ${_SERVER_DIR}
    ${_ROOT_DIR}
    ${_ROOT_DIR}/include
    ${_ROOT_DIR}/common
    ${_ROOT_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
    ${_ROOT_DIR}/vendor
    ${_ROOT_DIR}/vendor/nlohmann
)

target_link_libraries(joint-server PRIVATE premise-server-lib mtmd cpp-httplib llama-ui)
target_compile_definitions(joint-server PRIVATE
    LLAMA_SERVER_CONTEXT_HOOKS_INCLUDE=\"joint-retrieval-hooks.hpp\")
target_compile_features(joint-server PRIVATE cxx_std_17)
