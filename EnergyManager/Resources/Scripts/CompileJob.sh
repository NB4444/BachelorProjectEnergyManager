#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --constraint=GPU
#SBATCH -o out.%j
##SBATCH -e err.%j
#SBATCH --time=1:00:00

module purge
source /hpc/base/ctt/bin/setup_modules.sh
module load base-env
module load git/2.6.2
module load compiler/gnu/8.2.0
module load cmake/3.14.5
module load cuda/10.1
module load boost/1.70.0/intel

cd ../..

srun /hpc/base/ctt/packages/cmake/3.14.5/bin/cmake --build /home/xqbakker/EnergyManager/EnergyManager/cmake-build-debug-lenox --target EnergyManager-Library -- -j 12
