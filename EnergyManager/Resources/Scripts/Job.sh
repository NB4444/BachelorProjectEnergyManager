#!/bin/bash
#SBATCH -p gpu_short
#SBATCH -N 1
#SBATCH -t 00:10:00

test="MatrixMultiplyTest"
parameters="-p \"name=Matrix Multiply Test\" -p \"cpu=0\" -p \"gpu=0\" -p \"matrixAWidth=640\" -p \"matrixAHeight=640\" -p \"matrixBWidth=640\" -p \"matrixBHeight=640\""

executable="$HOME/ear_sara_collab/EnergyManager/build/EnergyManager"
database="$HOME/ear_sara_collab/EnergyManager/Resources/Test Results/database.sqlite"

module purge
module load 2019
module load foss/2018b
module load CUDA/10.1.243
module load SQLite/3.24.0-GCCcore-7.3.0
module load CMake/3.12.1-GCCcore-7.3.0

srun "$executable" -d "$database" -t "$test" "$parameters"