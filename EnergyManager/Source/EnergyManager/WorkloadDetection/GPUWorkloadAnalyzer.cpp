#include "./GPUWorkloadAnalyzer.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <utility>

namespace EnergyManager {
	namespace WorkloadDetection {
		std::shared_ptr<Workloads::Workload> GPUWorkloadAnalyzer::onAnalyzeWorkload() {
			// TODO
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("NOT IMPLEMENTED");
		}

		GPUWorkloadAnalyzer::GPUWorkloadAnalyzer(std::shared_ptr<Hardware::GPU> gpu) : gpu_(std::move(gpu)) {
		}
	}
}