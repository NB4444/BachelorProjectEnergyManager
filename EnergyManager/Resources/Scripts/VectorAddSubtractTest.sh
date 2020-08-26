#!/bin/bash

gpu=0
computeCount=50000

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
	--test "VectorAddSubtractTest" \
	--parameter "name=Vector Add Subtract Test" \
	--parameter "gpu=$gpu" \
	--parameter "computeCount=$computeCount"
