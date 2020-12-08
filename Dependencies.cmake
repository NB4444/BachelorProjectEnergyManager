include(ExternalProject)

# Configure pthreads
find_package(Threads REQUIRED)
message(STATUS "Threads libraries: ${CMAKE_THREAD_LIBS_INIT}")

# Set up CUDA
find_package(CUDA 10.1 REQUIRED)
set(CUDA_DIRECTORY "${CUDA_TOOLKIT_ROOT_DIR}")
set(CUDA_BINARY_DIRECTORY "${CUDA_DIRECTORY}/bin")
#set(CUDA_SAMPLES_DIRECTORY "${CUDA_DIRECTORY}/samples")
set(CUDA_SAMPLES_DIRECTORY "/home/xqbakker/cuda-10.1/samples")
if (NOT EXISTS ${CUDA_SAMPLES_DIRECTORY})
    set(CUDA_SAMPLES_DIRECTORY "/home/qub1/cuda-10.1/samples")
endif ()
set(CUDA_CUPTI_SAMPLES_DIRECTORY "/home/xqbakker/cuda-10.1/extras/CUPTI/samples")
if (NOT EXISTS ${CUDA_CUPTI_SAMPLES_DIRECTORY})
    set(CUDA_CUPTI_SAMPLES_DIRECTORY "/home/qub1/cuda-10.1/extras/CUPTI/samples")
endif ()
find_library(CUDA_LIBRARY cuda PATHS "${CUDA_DIRECTORY}/lib64/stubs" "${CUDA_DIRECTORY}/lib/stubs")
set(CUDA_NVML_LIBRARY "${CUDA_DIRECTORY}/lib64/stubs/libnvidia-ml.so")
set(CUDA_RUNTIME_LIBRARY "${CUDA_LIBRARIES}")
#set(CUDA_STATIC_RUNTIME_LIBRARY "${CUDA_DIRECTORY}/lib64/libcudart.so")
set(CUDA_STATIC_RUNTIME_LIBRARY "${CUDA_cudart_static_LIBRARY}")
set(CUDA_CUPTI_INCLUDE_DIRECTORY "${CUDA_DIRECTORY}/extras/CUPTI/include")
set(CUDA_CUPTI_LIBRARY_DIRECTORY "${CUDA_DIRECTORY}/extras/CUPTI/lib64")
set(CUDA_CUPTI_LIBRARY "${CUDA_cupti_LIBRARY}" "${CUDA_CUPTI_LIBRARY_DIRECTORY}/libnvperf_host.so" "${CUDA_CUPTI_LIBRARY_DIRECTORY}/libnvperf_target.so")
find_program(CUDA_COMPILER "nvcc" PATHS "${CUDA_BINARY_DIRECTORY}")
add_compile_definitions(CUDA_SAMPLES_DIRECTORY="${CUDA_SAMPLES_DIRECTORY}")
add_compile_definitions(CUDA_CUPTI_SAMPLES_DIRECTORY="${CUDA_CUPTI_SAMPLES_DIRECTORY}")
message(STATUS "CUDA CUPTI library directory: ${CUDA_CUPTI_LIBRARY_DIRECTORY}")
message(STATUS "CUDA CUPTI library: ${CUDA_CUPTI_LIBRARY}")
message(STATUS "CUDA CUPTI samples directory: ${CUDA_SAMPLES_DIRECTORY}")
message(STATUS "CUDA NVML library: ${CUDA_NVML_LIBRARY}")
message(STATUS "CUDA binary directory: ${CUDA_BINARY_DIRECTORY}")
message(STATUS "CUDA compiler: ${CUDA_COMPILER}")
message(STATUS "CUDA directory: ${CUDA_DIRECTORY}")
message(STATUS "CUDA library: ${CUDA_LIBRARY}")
message(STATUS "CUDA runtime library: ${CUDA_RUNTIME_LIBRARY}")
message(STATUS "CUDA samples directory: ${CUDA_SAMPLES_DIRECTORY}")
message(STATUS "CUDA static runtime library: ${CUDA_STATIC_RUNTIME_LIBRARY}")

# Enable CUDA
set(CMAKE_CUDA_COMPILER "${CUDA_COMPILER}")
message(STATUS "CUDA compiler: ${CUDA_COMPILER}")
set(CMAKE_CUDA_HOST_COMPILER "${GXX_COMPILER}")
enable_language(CUDA)

# Integrate CUPTI with the reporter
set(REPORTER_LIBRARY_DEPENDENCIES "${CUDA_cupti_LIBRARY}")
add_compile_definitions(REPORTER_LIBRARY_DEPENDENCIES="${REPORTER_LIBRARY_DEPENDENCIES}")
message(STATUS "Reporter library dependencies: ${REPORTER_LIBRARY_DEPENDENCIES}")

# Configure Nvidia Code Samples
set(NVIDIA_CODE_SAMPLES_DIRECTORY "/home/xqbakker/code-samples")
add_compile_definitions(NVIDIA_CODE_SAMPLES_DIRECTORY="${NVIDIA_CODE_SAMPLES_DIRECTORY}")
message(STATUS "NVIDIA code samples directory: ${NVIDIA_CODE_SAMPLES_DIRECTORY}")

# Configure Rodinia
set(RODINIA_DIRECTORY "/home/xqbakker/rodinia_3.1")
if (NOT EXISTS ${RODINIA_DIRECTORY})
    set(RODINIA_DIRECTORY "/home/qub1/rodinia_3.1")
endif ()
set(RODINIA_BINARY_DIRECTORY "${RODINIA_DIRECTORY}/bin")
set(RODINIA_DATA_DIRECTORY "${RODINIA_DIRECTORY}/data")
add_compile_definitions(RODINIA_DIRECTORY="${RODINIA_DIRECTORY}")
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

# Configure SLURM
find_program(SLURM_SACCT sacct)
find_program(SLURM_SALLOC salloc)
find_program(SLURM_SATTACH sattach)
find_program(SLURM_SBATCH sbatch)
find_program(SLURM_SRUN srun)
find_program(SLURM_SCONTROL scontrol)
add_compile_definitions(SLURM_SACCT="${SLURM_SACCT}")
add_compile_definitions(SLURM_SALLOC="${SLURM_SALLOC}")
add_compile_definitions(SLURM_SATTACH="${SLURM_SATTACH}")
add_compile_definitions(SLURM_SBATCH="${SLURM_SBATCH}")
add_compile_definitions(SLURM_SRUN="${SLURM_SRUN}")
add_compile_definitions(SLURM_SCONTROL="${SLURM_SCONTROL}")
message(STATUS "SLURM sacct: ${SLURM_SACCT}")
message(STATUS "SLURM salloc: ${SLURM_SALLOC}")
message(STATUS "SLURM sattach: ${SLURM_SATTACH}")
message(STATUS "SLURM sbatch: ${SLURM_SBATCH}")
message(STATUS "SLURM srun: ${SLURM_SRUN}")
message(STATUS "SLURM scontrol: ${SLURM_SCONTROL}")

# Configure EAR
set(EAR_DIRECTORY "/hpc/base/ctt/packages/EAR/ear")
find_library(EAR_LIBRARY ear.seq PATHS "${EAR_DIRECTORY}/lib")
find_library(EAR_LIBRARY_DAEMON earld "${EAR_DIRECTORY}/lib")
set(EAR_EACCT "${EAR_DIRECTORY}/bin/eacct")
add_compile_definitions(EAR_LIBRARY="${EAR_LIBRARY}")
add_compile_definitions(EAR_LIBRARY_DAEMON="${EAR_LIBRARY_DAEMON}")
add_compile_definitions(EAR_EACCT="${EAR_EACCT}")
message(STATUS "EAR directory: ${EAR_DIRECTORY}")
message(STATUS "EAR library: ${EAR_LIBRARY}")
message(STATUS "EAR library daemon: ${EAR_LIBRARY_DAEMON}")
message(STATUS "EAR eacct: ${EAR_EACCT}")