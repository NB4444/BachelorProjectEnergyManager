#include "./SyntheticGPUWorkload.hpp"

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stack>

namespace EnergyManager {
	namespace Testing {
		namespace Benchmarking {
			__global__ void add(const int* left, const int* right, int* result) {
				int index = blockDim.x * blockIdx.x + threadIdx.x;

				result[index] = left[index] + right[index];
			}

			__global__ void subtract(const int* left, const int* right, int* result) {
				int index = blockDim.x * blockIdx.x + threadIdx.x;

				result[index] = left[index] - right[index];
			}

			void SyntheticGPUWorkload::processOperation(const SyntheticGPUOperation& operation, const std::map<std::string, std::string>& parameters) {
				static std::vector<std::pair<int*, size_t>> hostVariables = {};
				static std::vector<std::pair<int*, size_t>> deviceVariables = {};

				auto getParameter = [&](const std::string& name, const std::string& defaultValue) {
					if(parameters.find(name) == parameters.end()) {
						return defaultValue;
					} else {
						return parameters.at(name);
					}
				};

				switch(operation) {
					case SyntheticGPUOperation::HOST_ALLOCATE: {
						size_t size = std::stoi(getParameter("size", "8"));

						int* hostVariable = (int*) malloc(size);
						hostVariables.push_back({ hostVariable, size });
						break;
					}
					case SyntheticGPUOperation::HOST_ASSIGN: {
						unsigned int count = std::stoi(getParameter("count", "1"));

						size_t hostVariableIndex = hostVariables.size() - 1;
						while(count > 0 && hostVariableIndex > 0) {
							int* hostVariable = hostVariables[hostVariableIndex].first;
							--hostVariableIndex;

							*hostVariable = std::rand();

							--count;
						}
						break;
					}
					case SyntheticGPUOperation::HOST_FREE: {
						int* hostVariable = hostVariables.back().first;
						hostVariables.pop_back();
						free(hostVariable);
						break;
					}
					case SyntheticGPUOperation::DEVICE_ALLOCATE: {
						size_t size = std::stoi(getParameter("size", "8"));

						int* deviceVariable = nullptr;
						ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMalloc((void**) &deviceVariable, size));
						deviceVariables.push_back({ deviceVariable, size });
						break;
					}
					case SyntheticGPUOperation::DEVICE_ASSIGN: {
						unsigned int count = std::stoi(getParameter("count", "1"));

						size_t deviceVariableIndex = deviceVariables.size() - 1;
						while(count > 0 && deviceVariableIndex > 0) {
							int* deviceVariable = deviceVariables[deviceVariableIndex].first;
							--deviceVariableIndex;

							*deviceVariable = std::rand();

							--count;
						}
						break;
					}
					case SyntheticGPUOperation::DEVICE_FREE: {
						int* deviceVariable = deviceVariables.back().first;
						deviceVariables.pop_back();
						ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaFree(deviceVariable));
						break;
					}
					case SyntheticGPUOperation::COPY_HOST_TO_DEVICE: {
						unsigned int count = std::stoi(getParameter("count", "1"));

						size_t hostVariableIndex = hostVariables.size() - 1;
						size_t deviceVariableIndex = deviceVariables.size() - 1;
						while(count > 0 && hostVariableIndex > 0 && deviceVariableIndex > 0) {
							int* hostVariable = hostVariables[hostVariableIndex].first;
							--hostVariableIndex;
							int* deviceVariable = deviceVariables[deviceVariableIndex].first;
							size_t deviceVariableSize = deviceVariables[deviceVariableIndex].second;
							--deviceVariableIndex;

							ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMemcpy(hostVariable, deviceVariable, deviceVariableSize, cudaMemcpyHostToDevice));

							--count;
						}
						break;
					}
					case SyntheticGPUOperation::COPY_DEVICE_TO_HOST: {
						unsigned int count = std::stoi(getParameter("count", "1"));

						size_t hostVariableIndex = hostVariables.size() - 1;
						size_t deviceVariableIndex = deviceVariables.size() - 1;
						while(count > 0 && hostVariableIndex > 0 && deviceVariableIndex > 0) {
							int* hostVariable = hostVariables[hostVariableIndex].first;
							size_t hostVariableSize = hostVariables[hostVariableIndex].second;
							--hostVariableIndex;
							int* deviceVariable = deviceVariables[deviceVariableIndex].first;
							--deviceVariableIndex;

							ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMemcpy(hostVariable, deviceVariable, hostVariableSize, cudaMemcpyDeviceToHost));

							--count;
						}
						break;
					}
					case SyntheticGPUOperation::DEVICE_ADD: {
						unsigned int count = std::stoi(getParameter("count", "1"));
						auto threadsPerBlock = std::stoi(getParameter("threadsPerBlock", "1"));
						auto blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
						int* leftVariable = deviceVariables.back().first;
						int* rightVariable = deviceVariables[deviceVariables.size() - 2].first;
						int* resultVariable = deviceVariables[deviceVariables.size() - 3].first;

						add<<<blocksPerGrid, threadsPerBlock>>>(leftVariable, rightVariable, resultVariable);
						ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaGetLastError());
						break;
					}
					case SyntheticGPUOperation::DEVICE_SUBTRACT: {
						unsigned int count = std::stoi(getParameter("count", "1"));
						auto threadsPerBlock = std::stoi(getParameter("threadsPerBlock", "1"));
						auto blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
						int* leftVariable = deviceVariables.back().first;
						int* rightVariable = deviceVariables[deviceVariables.size() - 2].first;
						int* resultVariable = deviceVariables[deviceVariables.size() - 3].first;

						subtract<<<blocksPerGrid, threadsPerBlock>>>(leftVariable, rightVariable, resultVariable);
						ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaGetLastError());
						break;
					}
					default: {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Unknown operation");
					}
				}
			}
		}
	}
}