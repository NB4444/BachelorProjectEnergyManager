#include "./WorkloadMonitor.hpp"

#include <string>

namespace Profiling {
	WorkloadMonitor::WorkloadMonitor(const GPUMonitor& gpuMonitor)
		: gpuMonitor_(gpuMonitor) {
	}

	std::map<std::string, std::string> WorkloadMonitor::onPoll() {
		// TODO: Detect workload type here
		std::string workloadName = "TODO";

		return {
			{ "workloadName", workloadName }
		};
	}
}