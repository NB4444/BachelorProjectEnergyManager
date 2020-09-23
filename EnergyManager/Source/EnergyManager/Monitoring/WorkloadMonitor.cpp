#include "./WorkloadMonitor.hpp"

namespace EnergyManager {
	namespace Monitoring {
		WorkloadMonitor::WorkloadMonitor(const std::shared_ptr<Hardware::GPU>& gpu) : Monitor("WorkloadMonitor"), gpu_(gpu) {
		}

		std::map<std::string, std::string> WorkloadMonitor::onPoll() {
			// TODO: Detect workload type here
			std::string workloadName = "TODO";

			return { { "workloadName", workloadName } };
		}
	}
}