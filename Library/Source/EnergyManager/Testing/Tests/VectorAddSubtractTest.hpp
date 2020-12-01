#pragma once

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/Tests/Test.hpp"

#include <device_launch_parameters.h>
#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			__global__ void vectorAdd(const int* A, const int* B, int* C, int N);

			__global__ void vectorSubtract(const int* A, const int* B, int* C, int N);

			/**
			 * The testing code was sampled from the `activity_trace_async` sample in CUDA 10.1's CUPTI library.
			 */
			class VectorAddSubtractTest : public EnergyManager::Testing::Tests::Test {
				/**
				 * The GPU to use.
				 */
				const std::shared_ptr<EnergyManager::Hardware::GPU>& gpu_;

				/**
				 * The amount of computations to perform.
				 */
				int computeCount_;

				/**
				 * Does one pass.
				 * @param stream The stream to useThe stream to use
				 */
				void doPass(cudaStream_t stream) const;

			protected:
				std::map<std::string, std::string> onTest() final;

			public:
				/**
				 * Creates a new VectorAddSubtractTest.
				 * @param gpu The GPU to use.
				 * @param computeCount The amount of computations to perform.
				 * @param monitors The monitors to use.
				 */
				VectorAddSubtractTest(
					const std::shared_ptr<EnergyManager::Hardware::GPU>& gpu,
					const unsigned int& computeCount,
					const std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>>& monitors = {});

				/**
				 * Creates a new VectorAddSubtractTest from command line arguments.
				 * @param arguments The command line arguments.
				 */
				explicit VectorAddSubtractTest(const std::map<std::string, std::string>& arguments);
			};
		}
	}
}