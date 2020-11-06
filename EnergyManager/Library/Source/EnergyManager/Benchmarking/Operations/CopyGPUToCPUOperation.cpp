#include "./CopyGPUToCPUOperation.hpp"

#include "EnergyManager/Hardware/GPU.hpp"

#include <cuda_runtime.h>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			void CopyGPUToCPUOperation::onRun() {
				auto count = count_;

				size_t hostVariableIndex = MemoryCPUOperation::variables_.size() - 1;
				size_t deviceVariableIndex = MemoryGPUOperation::variables_.size() - 1;
				while(count > 0 && hostVariableIndex > 0 && deviceVariableIndex > 0) {
					int* hostVariable = MemoryCPUOperation::variables_[hostVariableIndex].first;
					size_t hostVariableSize = MemoryCPUOperation::variables_[hostVariableIndex].second;
					--hostVariableIndex;
					int* deviceVariable = MemoryGPUOperation::variables_[deviceVariableIndex].first;
					--deviceVariableIndex;

					ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMemcpy(hostVariable, deviceVariable, hostVariableSize, cudaMemcpyDeviceToHost));

					--count;
				}
			}

			CopyGPUToCPUOperation::CopyGPUToCPUOperation(const unsigned int& count) : count_(count) {
			}
		}
	}
}