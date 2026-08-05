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

# Optional premise retrieval mode (--premise), built into llama-server
if (PremiseServer)
    set(_ROOT_DIR "${CMAKE_SOURCE_DIR}")

    add_library(llama-server-premise STATIC
        premise/premise.cpp
    )
    target_include_directories(llama-server-premise PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/premise
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${_ROOT_DIR}
        ${_ROOT_DIR}/include
        ${_ROOT_DIR}/common
        ${_ROOT_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
        ${_ROOT_DIR}/vendor
        ${_ROOT_DIR}/vendor/nlohmann
    )
    target_link_libraries(llama-server-premise PUBLIC llama llama-common)
    target_compile_features(llama-server-premise PUBLIC cxx_std_17)

    if (NOT TARGET faiss AND NOT TARGET faiss::faiss)
        find_package(faiss CONFIG QUIET)
    endif()
    if (NOT TARGET faiss AND NOT TARGET faiss::faiss)
        add_subdirectory(${_ROOT_DIR}/vendor/faiss ${CMAKE_BINARY_DIR}/vendor/faiss)
    endif()
    if (TARGET faiss)
        set(_premise_faiss_target faiss)
    elseif (TARGET faiss::faiss)
        set(_premise_faiss_target faiss::faiss)
    else()
        message(FATAL_ERROR "FAISS was found, but no known CMake target was exported")
    endif()
    target_link_libraries(llama-server-premise PUBLIC ${_premise_faiss_target})
    message(STATUS "llama-server: premise mode enabled (use --premise)")
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (TARGET llama-server-premise)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_PREMISE_SERVER)
    target_link_libraries(${TARGET} PRIVATE llama-server-premise)
endif()
