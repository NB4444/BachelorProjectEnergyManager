#!/bin/bash
#SBATCH --output log.%J
#SBATCH --error log.%J
#SBATCH --job-name EnergyManager-JobRunner
#SBATCH --account COLBSC
#SBATCH --partition standard
#SBATCH --time 8:00:00
#SBATCH --constraint V100_16GB&NUMGPU2&rack26&EDR
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive

module purge
source /hpc/base/ctt/bin/setup_modules.sh
module load base-env
module load git/2.6.2
module load compiler/intel/20.4
module load compiler/gnu/8.2.0
module load cmake/3.14.5
module load cuda/10.1
module load boost/1.67.0/impi
module load ear/ear

# Load the EAR library
export LD_PRELOAD=${EAR_INSTALL_PATH}/lib/libear.seq.so
export SLURM_HACK_LIBRARY_FILE=${EAR_INSTALL_PATH}/lib/libear.seq.so

srun \
  --job-name EnergyManager \
  --ear-verbose 1 \
  --ear-policy monitoring \
  "$@"
