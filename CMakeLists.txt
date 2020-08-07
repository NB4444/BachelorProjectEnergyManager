cmake_minimum_required(VERSION 3.8)

# Define the CUDA compiler to use
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(Project LANGUAGES CXX CUDA)

add_subdirectory(project)