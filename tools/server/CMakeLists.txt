include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-tools.cpp
    server-tools.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# premise retrieval mode (llama-server --premise), see premise/
option(LLAMA_SERVER_PREMISE "llama-server: build premise retrieval mode (--premise)" OFF)
if (LLAMA_SERVER_PREMISE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/vendor/sqlite ${CMAKE_BINARY_DIR}/vendor/sqlite)
    target_sources(${TARGET} PRIVATE premise/premise.cpp)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_PREMISE_SERVER)
    target_link_libraries(${TARGET} PRIVATE llama-sqlite3)

    add_executable(test-premise-index premise/test-index.cpp)
    target_link_libraries(test-premise-index PRIVATE llama-sqlite3 ggml)
    target_compile_features(test-premise-index PRIVATE cxx_std_17)
    if (LLAMA_BUILD_TESTS)
        add_test(NAME test-premise-index COMMAND test-premise-index)
    else()
        set_target_properties(test-premise-index PROPERTIES EXCLUDE_FROM_ALL ON)
    endif()
endif()
