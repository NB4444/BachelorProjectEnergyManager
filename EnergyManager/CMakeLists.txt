cmake_minimum_required(VERSION 3.14)

# Configure CUDA compiler
if (EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
elseif (EXISTS "/hpc/eb/compilerwrappers/compilers/nvcc")
    set(CMAKE_CUDA_COMPILER "/hpc/eb/compilerwrappers/compilers/nvcc")
else ()
    set(CMAKE_CUDA_COMPILER "nvcc")
endif ()

# Set up the project
project(EnergyManager LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set up directories
set(PROJECT_RESOURCES_DIRECTORY "${CMAKE_SOURCE_DIR}/Resources")
add_compile_definitions(PROJECT_RESOURCES_DIRECTORY="${PROJECT_RESOURCES_DIRECTORY}")
add_compile_definitions(PROJECT_DATABASE="/data/Cloud/Nextcloud/Education/Vrije Universiteit/Master Project/Project/EnergyManager/Resources/Test Results/database.sqlite")
add_compile_definitions(CUDA_SAMPLES_DIRECTORY="/usr/local/cuda-10.1/samples")
set(RODINIA_DIRECTORY="/home/qub1/rodinia_3.1")
add_compile_definitions(RODINIA_BIN_DIRECTORY="${RODINIA_DIRECTORY}/bin")
add_compile_definitions(RODINIA_DATA_DIRECTORY="${RODINIA_DIRECTORY}/data")

# Add targets
add_subdirectory(${CMAKE_SOURCE_DIR}/Library)
#add_subdirectory(Console)
add_subdirectory(${CMAKE_SOURCE_DIR}/Profiling)
add_subdirectory(${CMAKE_SOURCE_DIR}/Tests)
add_subdirectory(${CMAKE_SOURCE_DIR}/Training)
