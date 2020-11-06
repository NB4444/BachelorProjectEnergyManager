#include "./FreeGPUOperation.hpp"

#include "EnergyManager/Hardware/GPU.hpp"

#include <cuda_runtime.h>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			void FreeGPUOperation::onRun() {
				int* variable = variables_.back().first;
				variables_.pop_back();
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaFree(variable));
			}
		}
	}
}