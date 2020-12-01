#include "./AllocateFreeWorkloadTest.hpp"

#include "EnergyManager/Benchmarking/Workloads/AllocateFreeWorkload.hpp"
#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			AllocateFreeWorkloadTest::AllocateFreeWorkloadTest(const std::map<std::string, std::string>& arguments)
				: WorkloadTest(
					"Allocate Free Workload",
					std::make_shared<Benchmarking::Workloads::AllocateFreeWorkload>(
						EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--hostAllocations", 3000000),
						EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--hostSize", 1024),
						EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--deviceAllocations", 3000000),
						EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--deviceSize", 1024)),
					Monitoring::Monitors::Monitor::getMonitorsForAllDevices()) {
			}
		}
	}
}