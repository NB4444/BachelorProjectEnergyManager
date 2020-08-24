#!/bin/bash

gpu=0
computeCount=50000

executable="$HOME/Cloud/Nextcloud/Education/Vrije Universiteit/Master Project/Project/EnergyManager/cmake-build-debug/EnergyManager"
database="$HOME/Cloud/Nextcloud/Education/Vrije Universiteit/Master Project/Project/EnergyManager/Resources/Test Results/database.sqlite"

"$executable" \
	--database "$database" \
	--test "VectorAddSubtractTest" \
	--parameter "name=Vector Add Subtract Test" \
	--parameter "gpu=$gpu" \
	--parameter "computeCount=$computeCount"
