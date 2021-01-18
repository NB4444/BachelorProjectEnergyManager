include(ExternalProject)

# Configure SLURM
set(SLURM_ENABLED TRUE)
add_compile_definitions(SLURM_ENABLED="${SLURM_ENABLED}")
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
set(EAR_ENABLED TRUE)
add_compile_definitions(EAR_ENABLED="${EAR_ENABLED}")
#set(EAR_DIRECTORY "/home/xqbakker/EAR_GEN")
set(EAR_DIRECTORY "/hpc/base/ctt/packages/EAR/ear")
set(EAR_INCLUDE_DIRECTORY "${EAR_DIRECTORY}/include")
set(EAR_LIBRARY_DIRECTORY "${EAR_DIRECTORY}/lib")
add_compile_definitions(EAR_LIBRARY_DIRECTORY="${EAR_LIBRARY_DIRECTORY}")
find_library(EAR_LIBRARY ear.seq PATHS "${EAR_LIBRARY_DIRECTORY}")
find_library(EAR_LIBRARY_DAEMON earld "${EAR_LIBRARY_DIRECTORY}")
find_library(EAR_LIBRARY_API ear_api "${EAR_LIBRARY_DIRECTORY}")
set(EAR_EACCT "${EAR_DIRECTORY}/bin/eacct")
add_compile_definitions(EAR_LIBRARY="${EAR_LIBRARY}")
add_compile_definitions(EAR_LIBRARY_DAEMON="${EAR_LIBRARY_DAEMON}")
add_compile_definitions(EAR_EACCT="${EAR_EACCT}")
message(STATUS "EAR directory: ${EAR_DIRECTORY}")
message(STATUS "EAR include directory: ${EAR_INCLUDE_DIRECTORY}")
message(STATUS "EAR library directory: ${EAR_LIBRARY_DIRECTORY}")
message(STATUS "EAR library: ${EAR_LIBRARY}")
message(STATUS "EAR library daemon: ${EAR_LIBRARY_DAEMON}")
message(STATUS "EAR library API: ${EAR_LIBRARY_API}")
message(STATUS "EAR eacct: ${EAR_EACCT}")

# Configure MPI
#find_path(
#        MPI_INCLUDE_DIRECTORY
#        NAMES mpi.h
#        PATHS "/hpc/base/platform/mpi/8.1.1/include"
#)
#find_library(
#        MPI_LIBRARY
#        NAMES libmpi mpi
#        PATHS "/hpc/base/platform/mpi/8.1.1/lib/linux_amd64"
#)
find_package(MPI REQUIRED)
set(MPI_INCLUDE_DIRECTORY "${MPI_CXX_INCLUDE_DIRS}")
set(MPI_LIBRARY "${MPI_CXX_LIBRARIES}")
set(MPI_MPIEXEC "/home/qub1/openmpi-cuda/bin/mpiexec")
if (NOT EXISTS ${MPI_MPIEXEC})
    find_program(MPI_MPIEXEC mpiexec)
endif ()
add_compile_definitions(MPI_MPIEXEC="${MPI_MPIEXEC}")
message(STATUS "MPI include directory: ${MPI_INCLUDE_DIRECTORY}")
message(STATUS "MPI library: ${MPI_LIBRARY}")
message(STATUS "MPI mpiexec: ${MPI_MPIEXEC}")

# Configure SQLite
find_package(SQLite3 REQUIRED)
set(SQLITE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIRS}")
set(SQLITE_LIBRARY "${SQLite3_LIBRARIES}")
message(STATUS "SQLite3 include directories: ${SQLITE_INCLUDE_DIRECTORY}")
message(STATUS "SQLite3 libraries: ${SQLITE_LIBRARY}")

## Configure OpenMP
#find_package(OpenMP REQUIRED)
#if (OPENMP_FOUND)
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#    message(STATUS "OpenMP C flags: ${OpenMP_C_FLAGS}")
#    message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
#    message(STATUS "OpenMP EXE linker flags: ${OpenMP_EXE_LINKER_FLAGS}")
#endif ()
#message(STATUS "OpenMP libraries: ${OpenMP_LIBRARIES}")

# Configure Boost
set(Boost_NO_BOOST_CMAKE "on")
find_package(Boost 1.67 REQUIRED COMPONENTS math_c99 program_options unit_test_framework serialization filesystem)
#set(BOOST_ROOT_DIRECTORY "${Boost_INCLUDE_DIR}")
#if(NOT BOOST_ROOT_DIRECTORY AND EXISTS "/hpc/base/ctt/packages/boost/1.70.0/intel")
#    set(BOOST_ROOT_DIRECTORY "/hpc/base/ctt/packages/boost/1.70.0/intel")
#endif()
set(BOOST_INCLUDE_DIRECTORY "${Boost_INCLUDE_DIRS}")
set(BOOST_LIBRARIES_DIRECTORY "${Boost_LIBRARY_DIRS}")
set(BOOST_LIBRARY "${Boost_LIBRARIES}")
#set(BOOST_INCLUDE_DIRECTORY "${BOOST_ROOT_DIRECTORY}/include")
#set(BOOST_ROOT_DIRECTORY "${PROJECT_LIBRARIES_DIRECTORY}/boost")
#set(BOOST_INCLUDE_DIRECTORY "${PROJECT_LIBRARIES_DIRECTORY}/include")
#set(BOOST_LIBRARY "${PROJECT_LIBRARIES_DIRECTORY}/lib/libboost.so")
#message(STATUS "Boost root directory: ${BOOST_ROOT_DIRECTORY}")
message(STATUS "Boost include directory: ${BOOST_INCLUDE_DIRECTORY}")
message(STATUS "Boost libraries directory: ${BOOST_LIBRARIES_DIRECTORY}")
message(STATUS "Boost library: ${BOOST_LIBRARY}")
#ExternalProject_Add(
#        Boost
#        URL "https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz"
#        URL_MD5 "b50944c0c13f81ce2c006802a1186f5a"
#        CONFIGURE_COMMAND "./bootstrap.sh" "--prefix='${BOOST_ROOT_DIRECTORY}'" "--with-libraries=program_options,test,serialization"
#        BUILD_COMMAND "./b2" "install"
#        BUILD_IN_SOURCE true
#        INSTALL_COMMAND ""
#        INSTALL_DIR "${BOOST_ROOT_DIRECTORY}"
#