#pragma once

#include "EnergyManager/Testing/Benchmarking/SyntheticWorkload.hpp"

#include <cuda_runtime.h>

namespace EnergyManager {
	namespace Testing {
		namespace Benchmarking {
			enum class SyntheticGPUOperation {
				HOST_ALLOCATE,
				HOST_ASSIGN,
				HOST_FREE,
				DEVICE_ALLOCATE,
				DEVICE_ASSIGN,
				DEVICE_FREE,
				COPY_HOST_TO_DEVICE,
				COPY_DEVICE_TO_HOST,
				DEVICE_ADD,
				DEVICE_SUBTRACT
			};

			__global__ void add(const int* left, const int* right, int* result);

			__global__ void subtract(const int* left, const int* right, int* result);

			class SyntheticGPUWorkload : public SyntheticWorkload<SyntheticGPUOperation> {
			protected:
				void processOperation(const SyntheticGPUOperation& operation, const std::map<std::string, std::string>& parameters) override;

			public:
				using SyntheticWorkload<SyntheticGPUOperation>::SyntheticWorkload;
			};
		}
	}
}