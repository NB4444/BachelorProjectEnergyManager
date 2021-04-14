cmake_minimum_required(VERSION 3.13.4)

# Set up the project
project(EnergyManager LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set up dependencies
include(Dependencies/CUDA.cmake)

# Configure project paths
include(Paths.cmake)

# Add targets
add_subdirectory(Library)
add_subdirectory(Profiling)
add_subdirectory(Reporter)
add_subdirectory(Testing)
add_subdirectory(Visualizer)