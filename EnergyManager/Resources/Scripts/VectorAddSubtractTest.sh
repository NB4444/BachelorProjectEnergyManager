#!/bin/bash

gpu=0
computeCount=50000

projectDirectory="/home/qub1-creation/Cloud/Nextcloud/Education/Vrije Universiteit/Master Project/Project/EnergyManager"
buildDirectory="$projectDirectory/cmake-build-debug"
resourcesDirectory="$projectDirectory/Resources"
executable="$buildDirectory/EnergyManager"
database="$resourcesDirectory/Test Results/database.sqlite"

"$executable" \
	--database "$database" \
	--test "VectorAddSubtractTest" \
	--parameter "name=Vector Add Subtract Test" \
	--parameter "gpu=$gpu" \
	--parameter "computeCount=$computeCount"
