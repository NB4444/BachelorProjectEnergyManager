#include "./AllocateFreeWorkload.hpp"

#include "EnergyManager/Benchmarking/Operations/AllocateCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/AllocateGPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/FreeCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/FreeGPUOperation.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			AllocateFreeWorkload::AllocateFreeWorkload(const unsigned int& hostAllocations, const size_t& hostSize, const unsigned int& deviceAllocations, const size_t& deviceSize) {
				// Allocate host vectors
				for(unsigned int index = 0; index < hostAllocations; ++index) {
					addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AllocateCPUOperation>(hostSize));
				}

				// Allocate device vectors
				for(unsigned int index = 0; index < deviceAllocations; ++index) {
					addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AllocateGPUOperation>(deviceSize));
				}

				// Free device vectors
				for(unsigned int index = 0; index < deviceAllocations; ++index) {
					addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::FreeGPUOperation>());
				}

				// Free host vectors
				for(unsigned int index = 0; index < hostAllocations; ++index) {
					addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::FreeCPUOperation>());
				}
			}
		}
	}
}