#include "./ActiveInactiveWorkload.hpp"

#include <EnergyManager/Benchmarking/Operations/AddGPUOperation.hpp>
#include <EnergyManager/Benchmarking/Operations/AllocateCPUOperation.hpp>
#include <EnergyManager/Benchmarking/Operations/AllocateGPUOperation.hpp>
#include <EnergyManager/Benchmarking/Operations/AssignCPUOperation.hpp>
#include <EnergyManager/Benchmarking/Operations/CopyCPUToGPUOperation.hpp>
#include <EnergyManager/Benchmarking/Operations/CopyGPUToCPUOperation.hpp>
#include <EnergyManager/Benchmarking/Operations/FreeCPUOperation.hpp>
#include <EnergyManager/Benchmarking/Operations/FreeGPUOperation.hpp>
#include <EnergyManager/Benchmarking/Operations/SleepOperation.hpp>
#include <EnergyManager/Utility/Text.hpp>

namespace Workloads {
	ActiveInactiveWorkload::ActiveInactiveWorkload(const unsigned int& activeOperations, const std::chrono::system_clock::duration& inactivePeriod, const unsigned int& cycles) {
		for(unsigned int cycle = 0; cycle < cycles; ++cycle) {
			for(unsigned int operation = 0; operation < activeOperations; ++operation) {
				size_t size = 1024;

				// Allocate host input vectors
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AllocateCPUOperation>(size));
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AllocateCPUOperation>(size));
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AssignCPUOperation>(2));

				// Allocate device input vectors
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AllocateGPUOperation>(size));
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AllocateGPUOperation>(size));
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::CopyCPUToGPUOperation>(2));
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::FreeCPUOperation>());
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::FreeCPUOperation>());

				// Do vector add
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AllocateGPUOperation>(size));
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AddGPUOperation>(2, 32));

				// Copy results to host
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::AllocateCPUOperation>(size));
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::CopyGPUToCPUOperation>(1));
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::FreeGPUOperation>());
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::FreeCPUOperation>());

				// Free device vectors
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::FreeGPUOperation>());
				addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::FreeGPUOperation>());
			}

			// Sleep
			addOperation(std::make_shared<EnergyManager::Benchmarking::Operations::SleepOperation>(inactivePeriod));
		}
	}
}
