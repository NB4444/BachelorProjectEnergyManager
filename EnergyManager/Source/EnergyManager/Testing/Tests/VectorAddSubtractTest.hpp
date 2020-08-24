#pragma once

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/Tests/Test.hpp"

#include <string>

/**
 * The testing code was sampled from the `activity_trace_async` sample in CUDA 10.1's CUPTI library.
 */

namespace EnergyManager {
	namespace Testing {
		class TestResults;

		namespace Tests {
			__global__ void vectorAdd(const int* A, const int* B, int* C, int N);

			__global__ void vectorSubtract(const int* A, const int* B, int* C, int N);

			class VectorAddSubtractTest : public Test {
				const Hardware::GPU& gpu_;

				int computeCount_;

				void doPass(cudaStream_t stream) const;

			protected:
				std::map<std::string, std::string> onRun() override;

			public:
				VectorAddSubtractTest(const std::string& name, const Hardware::GPU& gpu, const int& computeCount);
			};
		}
	}
}