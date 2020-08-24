#!/bin/bash

gpu=0
computeCount=50000

executable="/home/qub1-creation/Cloud/Nextcloud/Education/Vrije Universiteit/Master Project/Project/EnergyManager/cmake-build-debug/EnergyManager"
database="/home/qub1-creation/Cloud/Nextcloud/Education/Vrije Universiteit/Master Project/Project/EnergyManager/Resources/Test Results/database.sqlite"

"$executable" \
	--database "$database" \
	--test "VectorAddSubtractTest" \
	--parameter "name=Vector Add Subtract Test" \
	--parameter "gpu=$gpu" \
	--parameter "computeCount=$computeCount"
