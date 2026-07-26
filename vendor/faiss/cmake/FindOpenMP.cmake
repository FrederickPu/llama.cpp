set(OpenMP_FOUND TRUE)
set(OpenMP_C_FOUND TRUE)
set(OpenMP_CXX_FOUND TRUE)
set(OpenMP_C_VERSION "0.0")
set(OpenMP_CXX_VERSION "0.0")

if (NOT TARGET OpenMP::OpenMP_C)
    add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)
    set_target_properties(OpenMP::OpenMP_C PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/..")
endif()

if (NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/..")
endif()
