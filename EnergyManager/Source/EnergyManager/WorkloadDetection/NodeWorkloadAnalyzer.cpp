#include "./NodeWorkloadAnalyzer.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <utility>

namespace EnergyManager {
	namespace WorkloadDetection {
		std::shared_ptr<Workloads::Workload> NodeWorkloadAnalyzer::onAnalyzeWorkload() {
			// TODO
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("NOT IMPLEMENTED");
		}

		NodeWorkloadAnalyzer::NodeWorkloadAnalyzer(std::shared_ptr<Hardware::Node> node) : node_(std::move(node)) {
		}
	}
}