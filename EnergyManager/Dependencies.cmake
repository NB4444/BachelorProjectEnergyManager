include(ExternalProject)

# Configure pthreads
find_package(Threads REQUIRED)
message(STATUS "Threads libraries: ${CMAKE_THREAD_LIBS_INIT}")

# Set up CUDA paths
find_package(CUDA 10.1 REQUIRED)
set(CUDA_DIRECTORY "${CUDA_TOOLKIT_ROOT_DIR}")
set(CUDA_BINARY_DIRECTORY "${CUDA_DIRECTORY}/bin")
#set(CUDA_SAMPLES_DIRECTORY "${CUDA_DIRECTORY}/samples")
set(CUDA_SAMPLES_DIRECTORY "/home/xqbakker/cuda-10.1/samples")
set(CUDA_NVML_LIBRARY "${CUDA_DIRECTORY}/lib64/stubs/libnvidia-ml.so")
set(CUDA_RUNTIME_LIBRARY "${CMAKE_CUDA_RUNTIME_LIBRARY}")
set(CUDA_STATIC_RUNTIME_LIBRARY "${CUDA_DIRECTORY}/lib64/libcudart.so")
set(CUDA_CUPTI_LIBRARY "${CUDA_cupti_LIBRARY}")
set(CUDA_CUPTI_INCLUDE_DIRECTORY "${CUDA_DIRECTORY}/extras/CUPTI/include")
find_program(CUDA_COMPILER "nvcc" PATHS "${CUDA_BINARY_DIRECTORY}")
add_compile_definitions(CUDA_SAMPLES_DIRECTORY="${CUDA_SAMPLES_DIRECTORY}")
message(STATUS "CUDA directory: ${CUDA_DIRECTORY}")
message(STATUS "CUDA binary directory: ${CUDA_BINARY_DIRECTORY}")
message(STATUS "CUDA NVML library: ${CUDA_NVML_LIBRARY}")
message(STATUS "CUDA runtime library: ${CUDA_RUNTIME_LIBRARY}")
message(STATUS "CUDA static runtime library: ${CUDA_STATIC_RUNTIME_LIBRARY}")
message(STATUS "CUDA CUPTI library: ${CUDA_CUPTI_LIBRARY}")
message(STATUS "CUDA compiler: ${CUDA_COMPILER}")
message(STATUS "CUDA samples directory: ${CUDA_SAMPLES_DIRECTORY}")

# Enable CUDA
set(CMAKE_CUDA_COMPILER "${CUDA_COMPILER}")
message(STATUS "CUDA compiler: ${CUDA_COMPILER}")
set(CMAKE_CUDA_HOST_COMPILER "${GXX_COMPILER}")
enable_language(CUDA)

# Configure Rodinia
set(RODINIA_DIRECTORY "/home/xqbakker/rodinia_3.1")
set(RODINIA_BINARY_DIRECTORY "${RODINIA_DIRECTORY}/bin")
set(RODINIA_DATA_DIRECTORY "${RODINIA_DIRECTORY}/data")
add_compile_definitions(RODINIA_BINARY_DIRECTORY="${RODINIA_BINARY_DIRECTORY}")
add_compile_definitions(RODINIA_DATA_DIRECTORY="${RODINIA_DATA_DIRECTORY}")
message(STATUS "Rodinia directory: ${RODINIA_DIRECTORY}")
message(STATUS "Rodinia binary directory: ${RODINIA_BINARY_DIRECTORY}")
message(STATUS "Rodinia data directory: ${RODINIA_DATA_DIRECTORY}")
#find_package(CUDA 8 REQUIRED)
#set(CUDA_8_DIRECTORY "/hpc/base/cuda/cuda-8.0")
#set(CUDA_8_SAMPLES_DIRECTORY "${CUDA_8_DIRECTORY}/samples")
#message(STATUS "CUDA 8 directory: ${CUDA_8_DIRECTORY}")
#message(STATUS "CUDA 8 samples directory: ${CUDA_8_SAMPLES_DIRECTORY}")
#ExternalProject_Add(
#        Rodinia
#        PREFIX "${RODINIA_DIRECTORY}"
#        URL "http://www.cs.virginia.edu/~kw5na/lava/Rodinia/Packages/Current/rodinia_3.1.tar.bz2"
#        URL_MD5 "047d983e62107972f217921aa0027b05"
#        CONFIGURE_COMMAND ""
#        BUILD_COMMAND cd cuda/bfs && make CC="'${CUDA_8_DIRECTORY}/bin/nvcc' '-ccbin=/hpc/base/ctt/packages/compiler/gnu/4.9.2/bin/g++'" CUDA_DIR="${CUDA_8_DIRECTORY}" SDK_DIR="${CUDA_8_SAMPLES_DIRECTORY}"
#        BUILD_IN_SOURCE true
#        INSTALL_COMMAND ""
#        INSTALL_DIR "${RODINIA_DIRECTORY}"
#)
