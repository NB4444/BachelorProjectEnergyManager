#include "./ActiveInactiveWorkloadTest.hpp"

#include "EnergyManager/Benchmarking/Workloads/ActiveInactiveWorkload.hpp"
#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			ActiveInactiveWorkloadTest::ActiveInactiveWorkloadTest(const std::map<std::string, std::string>& arguments)
				: WorkloadTest(
					"Active Inactive Workload",
					std::make_shared<Benchmarking::Workloads::ActiveInactiveWorkload>(
						EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--activeOperations", 5000),
						EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--inactivePeriod", std::chrono::milliseconds(2500)),
						EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--cycles", 5)),
					Monitoring::Monitors::Monitor::getMonitorsForAllDevices()) {
			}
		}
	}
}