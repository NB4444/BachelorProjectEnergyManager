#include "./AllocateFreeWorkload.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Benchmarking {
			AllocateFreeWorkload::AllocateFreeWorkload(const unsigned int& hostAllocations, const size_t& hostSize, const unsigned int& deviceAllocations, const size_t& deviceSize) {
				// Allocate host vectors
				for(unsigned int index = 0; index < hostAllocations; ++index) {
					addOperation(SyntheticGPUOperation::HOST_ALLOCATE, { { "size", std::to_string(hostSize) } });
				}

				// Allocate device vectors
				for(unsigned int index = 0; index < deviceAllocations; ++index) {
					addOperation(SyntheticGPUOperation::DEVICE_ALLOCATE, { { "size", std::to_string(deviceSize) } });
				}

				// Free device vectors
				for(unsigned int index = 0; index < deviceAllocations; ++index) {
					addOperation(SyntheticGPUOperation::DEVICE_FREE, {});
				}

				// Free host vectors
				for(unsigned int index = 0; index < hostAllocations; ++index) {
					addOperation(SyntheticGPUOperation::HOST_FREE, {});
				}
			}
		}
	}
}