# Find CUDA
find_package(CUDA 10.1 EXACT REQUIRED)

# Find the root directory
set(CUDA_DIRECTORY "${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA directory: ${CUDA_DIRECTORY}")

# Find the binary directory
set(CUDA_BINARY_DIRECTORIES "${CUDA_DIRECTORY}/bin")
message(STATUS "CUDA binary directories: ${CUDA_BINARY_DIRECTORIES}")

# Find the samples directory
set(CUDA_SAMPLES_DIRECTORY "/applications/cuda-samples")
add_compile_definitions(CUDA_SAMPLES_DIRECTORY="${CUDA_SAMPLES_DIRECTORY}")
message(STATUS "CUDA samples directory: ${CUDA_SAMPLES_DIRECTORY}")

# Find the CUPTI samples directory
set(CUDA_CUPTI_SAMPLES_DIRECTORY "${CUDA_DIRECTORY}/extras/CUPTI/samples")
add_compile_definitions(CUDA_CUPTI_SAMPLES_DIRECTORY="${CUDA_CUPTI_SAMPLES_DIRECTORY}")
message(STATUS "CUDA CUPTI samples directory: ${CUDA_SAMPLES_DIRECTORY}")

# Find the CUPTI include directory
set(CUDA_CUPTI_INCLUDE_DIRECTORIES "${CUDA_DIRECTORY}/extras/CUPTI/include")
message(STATUS "CUDA CUPTI include directories: ${CUDA_CUPTI_INCLUDE_DIRECTORIES}")

# Find the CUPTI library directory
set(CUDA_CUPTI_LIBRARY_DIRECTORIES "${CUDA_DIRECTORY}/extras/CUPTI/lib64")
message(STATUS "CUDA CUPTI library directories: ${CUDA_CUPTI_LIBRARY_DIRECTORIES}")

# Find the CUPTI library
set(CUDA_CUPTI_LIBRARIES "${CUDA_cupti_LIBRARY}" "${CUDA_CUPTI_LIBRARY_DIRECTORIES}/libnvperf_host.so" "${CUDA_CUPTI_LIBRARY_DIRECTORIES}/libnvperf_target.so")
message(STATUS "CUDA CUPTI libraries: ${CUDA_CUPTI_LIBRARIES}")

# Find the compiler
find_program(CUDA_COMPILER "nvcc" PATHS "${CUDA_BINARY_DIRECTORIES}")
message(STATUS "CUDA compiler: ${CUDA_COMPILER}")

# Enable CUDA
set(CMAKE_CUDA_COMPILER "${CUDA_COMPILER}")
set(CMAKE_CUDA_HOST_COMPILER "${GXX_COMPILER}")
enable_language(CUDA)

# Integrate CUPTI with the reporter
set(REPORTER_LIBRARY_DEPENDENCIES "${CUDA_cupti_LIBRARY}")
add_compile_definitions(REPORTER_LIBRARY_DEPENDENCIES="${REPORTER_LIBRARY_DEPENDENCIES}")
message(STATUS "Reporter library dependencies: ${REPORTER_LIBRARY_DEPENDENCIES}")