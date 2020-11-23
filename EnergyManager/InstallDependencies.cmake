include(ExternalProject)

# Configure Rodinia
ExternalProject_Add(
        Rodinia
        URL "http://www.cs.virginia.edu/~kw5na/lava/Rodinia/Packages/Current/rodinia_3.1.tar.bz2"
        URL_MD5 "047d983e62107972f217921aa0027b05"
        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'"
)
set(RODINIA_DIRECTORY "${PROJECT_LIBRARIES_DIRECTORY}/rodinia_3.1")
set(RODINIA_BINARY_DIRECTORY "${RODINIA_DIRECTORY}/bin")
set(RODINIA_DATA_DIRECTORY "${RODINIA_DIRECTORY}/data")
add_compile_definitions(RODINIA_BINARY_DIRECTORY="${RODINIA_BINARY_DIRECTORY}")
add_compile_definitions(RODINIA_DATA_DIRECTORY="${RODINIA_DATA_DIRECTORY}")
message(STATUS "Rodinia directory: ${RODINIA_DIRECTORY}")
message(STATUS "Rodinia binary directory: ${RODINIA_BINARY_DIRECTORY}")
message(STATUS "Rodinia data directory: ${RODINIA_DATA_DIRECTORY}")

# Configure pthreads
find_package(Threads REQUIRED)
message(STATUS "Threads libraries: ${CMAKE_THREAD_LIBS_INIT}")

# Set up CUDA paths
find_package(CUDA 10.1 REQUIRED)
set(CUDA_DIRECTORY "${CUDA_TOOLKIT_ROOT_DIR}")
set(CUDA_BINARY_DIRECTORY "${CUDA_DIRECTORY}/bin")
set(CUDA_SAMPLES_DIRECTORY "${CUDA_DIRECTORY}/samples")
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