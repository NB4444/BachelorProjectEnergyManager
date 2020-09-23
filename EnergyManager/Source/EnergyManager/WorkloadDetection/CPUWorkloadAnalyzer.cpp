#include "./CPUWorkloadAnalyzer.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <utility>

namespace EnergyManager {
	namespace WorkloadDetection {
		std::shared_ptr<Workloads::Workload> CPUWorkloadAnalyzer::onAnalyzeWorkload() {
			// TODO
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("NOT IMPLEMENTED");
		}

		CPUWorkloadAnalyzer::CPUWorkloadAnalyzer(std::shared_ptr<Hardware::CPU> cpu) : cpu_(std::move(cpu)) {
		}
	}
}