#!/bin/bash

cpu=0
gpu=0
sizeMultiplier=25
matrixAWidth=$((32 * $sizeMultiplier))
matrixAHeight=$((32 * $sizeMultiplier))
matrixBWidth=$((32 * $sizeMultiplier))
matrixBHeight=$((32 * $sizeMultiplier))

projectDirectory="/home/qub1-creation/Cloud/Nextcloud/Education/Vrije Universiteit/Master Project/Project/EnergyManager"
buildDirectory="$projectDirectory/cmake-build-debug"
resourcesDirectory="$projectDirectory/Resources"
executable="$buildDirectory/EnergyManager"
database="$resourcesDirectory/Test Results/database.sqlite"

"$executable" \
	--database "$database" \
	--test "MatrixMultiplyTest" \
	--parameter "name=Matrix Multiply Test" \
	--parameter "cpu=$cpu" \
	--parameter "gpu=$gpu" \
	--parameter "matrixAWidth=$matrixAWidth" \
	--parameter "matrixAHeight=$matrixAHeight" \
	--parameter "matrixBWidth=$matrixBWidth" \
	--parameter "matrixBHeight=$matrixBHeight"
