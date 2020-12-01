#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

#include <device_launch_parameters.h>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * Adds two variables together and stores the result.
			 * @param left The left hand operand variable.
			 * @param right The right hand operand variable.
			 * @param result The variable to hold the result.
			 */
			__global__ void add(const int* left, const int* right, int* result);

			/**
			 * An add Operation that runs on the GPU.
			 */
			class AddGPUOperation : public MemoryGPUOperation {
				/**
				 * The amount of variables to add.
				 */
				unsigned int count_;

				/**
				 * The amount of threads in a block on the GPU.
				 */
				unsigned int threadsPerBlock_;

			protected:
				void onRun() final;

			public:
				/**
				 * Creates a new add Operation that runs on the GPU.
				 * @param count The amount of variables to add.
				 * @param threadsPerBlock The amount of threads in a block on the GPU.
				 */
				explicit AddGPUOperation(const unsigned int& count = 1, const unsigned int& threadsPerBlock = 1);
			};
		}
	}
}