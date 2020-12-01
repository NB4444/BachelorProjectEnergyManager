#include "./VectorAddWorkloadTest.hpp"

#include "EnergyManager/Benchmarking/Workloads/VectorAddWorkload.hpp"
#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			VectorAddWorkloadTest::VectorAddWorkloadTest(const std::map<std::string, std::string>& arguments)
				: WorkloadTest(
					"Vector Add Workload",
					std::make_shared<Benchmarking::Workloads::VectorAddWorkload>(EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--size", 1024000000)),
					Monitoring::Monitors::Monitor::getMonitorsForAllDevices()) {
			}
		}
	}
}