cmake_minimum_required(VERSION 3.14.5)

# Set up GCC paths
find_program(GXX_COMPILER "g++-8")
if (NOT GXX_COMPILER AND EXISTS "/hpc/base/ctt/packages/compiler/gnu/8.2.0/bin/g++")
    set(GXX_COMPILER "/hpc/base/ctt/packages/compiler/gnu/8.2.0/bin/g++")
endif ()
if (NOT GXX_COMPILER)
    find_program(GXX_COMPILER "g++")
endif ()
find_program(GCC_COMPILER "gcc-8")
if (NOT GCC_COMPILER AND EXISTS "/hpc/base/ctt/packages/compiler/gnu/8.2.0/bin/gcc")
    set(GCC_COMPILER "/hpc/base/ctt/packages/compiler/gnu/8.2.0/bin/gcc")
endif ()
if (NOT GCC_COMPILER)
    find_program(GCC_COMPILER "gcc")
endif ()
find_program(GFC_COMPILER "gfortran-8")
if (NOT GFC_COMPILER AND EXISTS "/hpc/base/ctt/packages/compiler/gnu/8.2.0/bin/gfortran")
    set(GFC_COMPILER "/hpc/base/ctt/packages/compiler/gnu/8.2.0/bin/gfortran")
endif ()
if (NOT GFC_COMPILER)
    find_program(GFC_COMPILER "gfortran")
endif ()
message(STATUS "GCC compiler: ${GCC_COMPILER}")
message(STATUS "GXX compiler: ${GXX_COMPILER}")
message(STATUS "GFC compiler: ${GFC_COMPILER}")

# Set up GCC
set(CMAKE_C_COMPILER "${GCC_COMPILER}")
set(CMAKE_CXX_COMPILER "${GXX_COMPILER}")
set(CMAKE_Fortran_COMPILER "${GFC_COMPILER}")

# Set up the project
project(EnergyManager LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set up dependencies
include(InstallDependencies.cmake)

# Set up project paths
set(PROJECT_RESOURCES_DIRECTORY "${CMAKE_SOURCE_DIR}/Resources")
set(PROJECT_LIBRARIES_DIRECTORY "${CMAKE_BINARY_DIR}/Installations")
set(PROJECT_DATABASE "~/Test Results/database.sqlite")
add_compile_definitions(PROJECT_RESOURCES_DIRECTORY="${PROJECT_RESOURCES_DIRECTORY}")
add_compile_definitions(PROJECT_DATABASE="${PROJECT_DATABASE}")
include_directories(${PROJECT_LIBRARIES_DIRECTORY}/include)
message(STATUS "Project resources directory: ${PROJECT_RESOURCES_DIRECTORY}")
message(STATUS "Project libraries directory: ${PROJECT_LIBRARIES_DIRECTORY}")
message(STATUS "Project library installations directory: ${PROJECT_LIBRARY_INSTALLATIONS_DIRECTORY}")
message(STATUS "Project database: ${PROJECT_DATABASE}")

# Add targets
add_subdirectory(${CMAKE_SOURCE_DIR}/Library)
add_subdirectory(${CMAKE_SOURCE_DIR}/Profiling)
add_subdirectory(${CMAKE_SOURCE_DIR}/Tests)
add_subdirectory(${CMAKE_SOURCE_DIR}/Training)