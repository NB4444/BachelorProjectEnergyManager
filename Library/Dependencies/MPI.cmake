# Find mpiexec
set(MPI_MPIEXEC "$ENV{HOME}/openmpi-cuda/bin/mpiexec")
if (NOT EXISTS ${MPI_MPIEXEC})
    find_program(MPI_MPIEXEC mpiexec REQUIRED)
endif ()
add_compile_definitions(MPI_MPIEXEC="${MPI_MPIEXEC}")
message(STATUS "MPI mpiexec: ${MPI_MPIEXEC}")