# Find mpiexec
find_program(MPI_MPIEXEC mpiexec REQUIRED)
if (NOT EXISTS ${MPI_MPIEXEC})
	set(MPI_MPIEXEC "$ENV{HOME}/openmpi-cuda/bin/mpiexec")
endif ()
add_compile_definitions(MPI_MPIEXEC="${MPI_MPIEXEC}")
message(STATUS "MPI mpiexec: ${MPI_MPIEXEC}")