#!/bin/bash

cpu=0
gpu=0
sizeMultiplier=25
matrixAWidth=$((32 * $sizeMultiplier))
matrixAHeight=$((32 * $sizeMultiplier))
matrixBWidth=$((32 * $sizeMultiplier))
matrixBHeight=$((32 * $sizeMultiplier))

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
projectDirectory="$scriptDirectory/../.."
buildDirectory="$projectDirectory/cmake-build-default"
resourcesDirectory="$projectDirectory/Resources"
executable="$buildDirectory/EnergyManager"
database="$resourcesDirectory/Test Results/database.sqlite"

cd "$scriptDirectory"
./Build.sh

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
