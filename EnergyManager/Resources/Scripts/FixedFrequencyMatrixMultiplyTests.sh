#!/bin/bash

cpus=0
gpu=0
testSegments=4
sizeMultiplier=100
cpuMaximumCoreClockRate=3889646848
gpuMaximumCoreClockRate=2370000000
matrixAWidth=$((32 * $sizeMultiplier))
matrixAHeight=$((32 * $sizeMultiplier))
matrixBWidth=$((32 * $sizeMultiplier))
matrixBHeight=$((32 * $sizeMultiplier))
pollingInterval=500
cpuMonitorPollingInterval=$pollingInterval
gpuMonitorPollingInterval=$pollingInterval
nodeMonitorPollingInterval=$pollingInterval
applicationMonitorPollingInterval=$pollingInterval

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
projectDirectory="$scriptDirectory/../.."
buildDirectory="$projectDirectory/cmake-build-default"
resourcesDirectory="$projectDirectory/Resources"
executable="$buildDirectory/EnergyManager"
database="$resourcesDirectory/Test Results/database.sqlite"

cpuClockRatePerSegment=$(($cpuMaximumCoreClockRate / $testSegments))
gpuClockRatePerSegment=$(($gpuMaximumCoreClockRate / $testSegments))

cd "$scriptDirectory"
./Build.sh

for ((segmentIndex = 0; segmentIndex < $testSegments; ++segmentIndex)); do
	minimumCPUFrequency=$(($segmentIndex * $cpuClockRatePerSegment))
	maximumCPUFrequency=$((($segmentIndex + 1) * $cpuClockRatePerSegment))
	minimumGPUFrequency=$(($segmentIndex * $gpuClockRatePerSegment))
	maximumGPUFrequency=$((($segmentIndex + 1) * $gpuClockRatePerSegment))

	"$executable" \
		--database "$database" \
		--test "FixedFrequencyMatrixMultiplyTest" \
		--parameter "name=Fixed Frequency Matrix Multiply Test (CPU $minimumCPUFrequency-$maximumCPUFrequency | GPU $minimumGPUFrequency-$maximumGPUFrequency)" \
		--parameter "cpus=$cpus" \
		--parameter "gpu=$gpu" \
		--parameter "matrixAWidth=$matrixAWidth" \
		--parameter "matrixAHeight=$matrixAHeight" \
		--parameter "matrixBWidth=$matrixBWidth" \
		--parameter "matrixBHeight=$matrixBHeight" \
		--parameter "minimumCPUFrequency=$minimumCPUFrequency" \
		--parameter "maximumCPUFrequency=$maximumCPUFrequency" \
		--parameter "minimumGPUFrequency=$minimumGPUFrequency" \
		--parameter "maximumGPUFrequency=$maximumGPUFrequency" \
		--parameter "applicationMonitorPollingInterval=$applicationMonitorPollingInterval" \
		--monitor "CPUMonitor" \
		--parameter "pollingInterval=$cpuMonitorPollingInterval" \
		--parameter "cpu=0" \
		--monitor "GPUMonitor" \
		--parameter "pollingInterval=$gpuMonitorPollingInterval" \
		--parameter "gpu=$gpu" \
		--monitor "NodeMonitor" \
		--parameter "pollingInterval=$nodeMonitorPollingInterval"
done
