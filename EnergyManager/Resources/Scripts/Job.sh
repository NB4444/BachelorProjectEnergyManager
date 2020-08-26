#!/bin/bash
#SBATCH -p gpu_short
#SBATCH -N 1
#SBATCH -t 00:10:00

script="$1"
scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

module purge
module load 2019
module load foss/2018b
module load CUDA/10.1.243
module load SQLite/3.24.0-GCCcore-7.3.0
module load CMake/3.12.1-GCCcore-7.3.0

cd "$scriptDirectory"
./Build.sh

srun "$scriptDirectory/$script"
