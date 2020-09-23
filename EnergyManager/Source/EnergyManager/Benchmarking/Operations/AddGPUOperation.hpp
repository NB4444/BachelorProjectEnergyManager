#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

#include <device_launch_parameters.h>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			__global__ void add(const int* left, const int* right, int* result);

			class AddGPUOperation : public MemoryGPUOperation {
				unsigned int count_;

				unsigned int threadsPerBlock_;

			protected:
				void onRun() override;

			public:
				AddGPUOperation(const unsigned int& count = 1, const unsigned int& threadsPerBlock = 1);
			};
		}
	}
}