#include "./AllocateGPUOperation.hpp"

#include "EnergyManager/Hardware/GPU.hpp"

#include <cuda_runtime.h>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			void AllocateGPUOperation::onRun() {
				int* variable = nullptr;
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMalloc((void**) &variable, size_));
				variables_.emplace_back(variable, size_);
			}

			AllocateGPUOperation::AllocateGPUOperation(const size_t& size) : size_(size) {
			}
		}
	}
}