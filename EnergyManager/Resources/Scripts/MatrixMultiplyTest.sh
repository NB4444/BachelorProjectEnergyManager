#!/bin/bash

cpu=0
gpu=0
sizeMultiplier=25
matrixAWidth=$((32 * $sizeMultiplier))
matrixAHeight=$((32 * $sizeMultiplier))
matrixBWidth=$((32 * $sizeMultiplier))
matrixBHeight=$((32 * $sizeMultiplier))

executable="$HOME/Cloud/Nextcloud/Education/Vrije Universiteit/Master Project/Project/EnergyManager/cmake-build-debug/EnergyManager"
database="$HOME/Cloud/Nextcloud/Education/Vrije Universiteit/Master Project/Project/EnergyManager/Resources/Test Results/database.sqlite"

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
