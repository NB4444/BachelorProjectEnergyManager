cmake_minimum_required(VERSION 3.13.4)

# Set up the project
project(EnergyManager LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set up base dependencies
include(Dependencies/CUDA.cmake)

# Configure project paths
include(Paths.cmake)

# Set up dependencies
include(Dependencies/Boost.cmake)
include(Dependencies/EAR.cmake)
include(Dependencies/MPI.cmake)
include(Dependencies/NVIDIACodeSamples.cmake)
include(Dependencies/Rodinia.cmake)
include(Dependencies/SLURM.cmake)
include(Dependencies/SQLite.cmake)
include(Dependencies/Threads.cmake)
set(CMAKE_BUILD_TYPE Debug)
# Add targets
add_subdirectory(Example)
add_subdirectory(Library)
add_subdirectory(Profiling)
add_subdirectory(Reporter)
add_subdirectory(Testing)
add_subdirectory(Visualizer)
