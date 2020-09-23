#!/bin/bash

cpu=0
gpu=0
hostAllocations=1000000
hostSize=1024
deviceAllocations=3000000
deviceSize=1024

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
projectDirectory="$scriptDirectory/../.."
buildDirectory="$projectDirectory/cmake-build-default"
resourcesDirectory="$projectDirectory/Resources"
executable="$buildDirectory/EnergyManager"
database="$resourcesDirectory/Test Results/database.sqlite"
workload="AllocateFreeWorkload"

cd "$scriptDirectory"
./Build.sh

"$executable" \
	--database "$database" \
	--test "SyntheticGPUWorkloadTest" \
	--parameter "name=SyntheticGPUWorkloadTest ($workload)" \
	--parameter "workload=$workload" \
	--parameter "cpu=$cpu" \
	--parameter "gpu=$gpu" \
	--parameter "hostAllocations=$hostAllocations" \
	--parameter "hostSize=$hostSize" \
	--parameter "deviceAllocations=$deviceAllocations" \
	--parameter "deviceSize=$deviceSize"
