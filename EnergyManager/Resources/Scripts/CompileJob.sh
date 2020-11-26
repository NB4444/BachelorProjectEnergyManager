#!/bin/bash
#SBATCH --job-name=EnergyManager-Compile
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --constraint=2666MHz&GPU&V100_16GB
#SBATCH --output log.%J.out
#SBATCH --error log.%J.err
#SBATCH --time=1:00:00

module purge
source /hpc/base/ctt/bin/setup_modules.sh
module load base-env
module load git/2.6.2
module load compiler/intel/20.4
module load compiler/gnu/8.2.0
module load cmake/3.14.5
module load cuda/10.1
module load boost/1.67.0/impi

srun /hpc/base/ctt/packages/cmake/3.14.5/bin/cmake --build /home/xqbakker/EnergyManager/EnergyManager/cmake-build-debug-lenox --target EnergyManager-Library -- -j 12
