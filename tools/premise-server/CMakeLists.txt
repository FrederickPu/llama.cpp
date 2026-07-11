# tools/premise-server/CMakeLists.txt

set(_SERVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../server")
set(_ROOT_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_library(premise-server-lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/premise-init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/premise-retrieval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/premise-http.cpp
)

target_include_directories(premise-server-lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_SERVER_DIR}
    ${_ROOT_DIR}
    ${_ROOT_DIR}/include
    ${_ROOT_DIR}/common
    ${_ROOT_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
    ${_ROOT_DIR}/vendor
    ${_ROOT_DIR}/vendor/nlohmann
)

target_link_libraries(premise-server-lib PUBLIC llama llama-common)
target_compile_features(premise-server-lib PUBLIC cxx_std_17)

add_executable(premise-server
    ${CMAKE_CURRENT_SOURCE_DIR}/premise-server.cpp
    ${_SERVER_DIR}/server-context.cpp
    ${_SERVER_DIR}/server-http.cpp
    ${_SERVER_DIR}/server-chat.cpp
    ${_SERVER_DIR}/server-task.cpp
    ${_SERVER_DIR}/server-queue.cpp
    ${_SERVER_DIR}/server-common.cpp
    ${_SERVER_DIR}/server-tools.cpp
    ${_SERVER_DIR}/server-models.cpp
)

target_include_directories(premise-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
)

target_link_libraries(premise-server PRIVATE premise-server-lib mtmd cpp-httplib llama-ui)
target_compile_features(premise-server PRIVATE cxx_std_17)
