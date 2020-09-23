#!/bin/bash

cpu=0
gpu=0
activeOperations=5000
inactivePeriod=2500
cycles=5

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
projectDirectory="$scriptDirectory/../.."
buildDirectory="$projectDirectory/cmake-build-default"
resourcesDirectory="$projectDirectory/Resources"
executable="$buildDirectory/EnergyManager"
database="$resourcesDirectory/Test Results/database.sqlite"
workload="ActiveInactiveWorkload"

cd "$scriptDirectory"
./Build.sh

"$executable" \
	--database "$database" \
	--test "SyntheticGPUWorkloadTest" \
	--parameter "name=SyntheticGPUWorkloadTest ($workload)" \
	--parameter "workload=$workload" \
	--parameter "cpu=$cpu" \
	--parameter "gpu=$gpu" \
	--parameter "activeOperations=$activeOperations" \
	--parameter "inactivePeriod=$inactivePeriod" \
	--parameter "cycles=$cycles"
