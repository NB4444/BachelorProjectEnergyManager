#include "./AddGPUOperation.hpp"

#include "EnergyManager/Hardware/GPU.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			__global__ void add(const int* left, const int* right, int* result) {
				unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

				result[index] = left[index] + right[index];
			}

			void AddGPUOperation::onRun() {
				auto blocksPerGrid = (count_ + threadsPerBlock_ - 1) / threadsPerBlock_;
				int* leftVariable = variables_[variables_.size() - 3].first;
				int* rightVariable = variables_[variables_.size() - 2].first;
				int* resultVariable = variables_.back().first;

				add<<<blocksPerGrid, threadsPerBlock_>>>(leftVariable, rightVariable, resultVariable);
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaGetLastError());
			}

			AddGPUOperation::AddGPUOperation(const unsigned int& count, const unsigned int& threadsPerBlock) : count_(count), threadsPerBlock_(threadsPerBlock) {
			}
		}
	}
}