#pragma once

#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Testing/Tests/Test.hpp>
#include <string>

/**
 * The testing code was sampled from the `activity_trace_async` sample in CUDA 10.1's CUPTI library.
 */

namespace Tests {
	__global__ void vectorAdd(const int* A, const int* B, int* C, int N);

	__global__ void vectorSubtract(const int* A, const int* B, int* C, int N);

	class VectorAddSubtractTest : public EnergyManager::Testing::Tests::Test {
		const std::shared_ptr<EnergyManager::Hardware::GPU>& gpu_;

		int computeCount_;

		void doPass(cudaStream_t stream) const;

	protected:
		std::map<std::string, std::string> onTest() final;

	public:
		VectorAddSubtractTest(
			const std::string& name,
			const std::shared_ptr<EnergyManager::Hardware::GPU>& gpu,
			const unsigned int& computeCount,
			const std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>>& monitors = {});
	};
}