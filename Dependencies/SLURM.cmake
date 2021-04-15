# Find sbatch
find_program(SLURM_SBATCH sbatch)
add_compile_definitions(SLURM_SBATCH="${SLURM_SBATCH}")
message(STATUS "SLURM sbatch: ${SLURM_SBATCH}")

# Find scontrol
find_program(SLURM_SCONTROL scontrol)
add_compile_definitions(SLURM_SCONTROL="${SLURM_SCONTROL}")
message(STATUS "SLURM scontrol: ${SLURM_SCONTROL}")

# Enable SLURM if available
if (EXISTS ${SLURM_SBATCH})
	message(STATUS "SLURM found, enabling SLURM functionality")
	set(SLURM_ENABLED TRUE)
	add_compile_definitions(SLURM_ENABLED="${SLURM_ENABLED}")
else ()
	message(STATUS "SLURM not found, disabling SLURM functionality")
endif ()