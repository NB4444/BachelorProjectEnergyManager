#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

#include <device_launch_parameters.h>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * Subtracts a variable from another and stores the result.
			 * @param left The left hand operand variable.
			 * @param right The right hand operand variable.
			 * @param result The variable to hold the result.
			 */
			__global__ void subtract(const int* left, const int* right, int* result);

			/**
			 * A subtract Operation that runs on the GPU.
			 */
			class SubtractGPUOperation : public MemoryGPUOperation {
				/**
				 * The amount of variables to subtract.
				 */
				unsigned int count_;

				/**
				 * The amount of threads in a block on the GPU.
				 */
				unsigned int threadsPerBlock_;

			protected:
				void onRun() override;

			public:
				/**
				 * Creates a new subtract Operation that runs on the GPU.
				 * @param count The amount of variables to subtract.
				 * @param threadsPerBlock The amount of threads in a block on the GPU.
				 */
				explicit SubtractGPUOperation(const unsigned int& count = 1, const unsigned int& threadsPerBlock = 1);
			};
		}
	}
}