cmake_minimum_required(VERSION 3.14.5)

include(Compiler.cmake)

# Set up the project
project(EnergyManager LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(Paths.cmake)
include(Dependencies.cmake)

# Add targets
add_subdirectory(${CMAKE_SOURCE_DIR}/Library)
add_subdirectory(${CMAKE_SOURCE_DIR}/Profiling)
add_subdirectory(${CMAKE_SOURCE_DIR}/Tests)
add_subdirectory(${CMAKE_SOURCE_DIR}/Training)