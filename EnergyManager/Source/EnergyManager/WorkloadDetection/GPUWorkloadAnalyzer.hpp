#pragma once

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/WorkloadDetection/WorkloadAnalyzer.hpp"

namespace EnergyManager {
	namespace WorkloadDetection {
		class GPUWorkloadAnalyzer : public WorkloadAnalyzer {
			std::shared_ptr<Hardware::GPU> gpu_;

		protected:
			std::shared_ptr<Workloads::Workload> onAnalyzeWorkload() override;

		public:
			GPUWorkloadAnalyzer(std::shared_ptr<Hardware::GPU> gpu);
		};
	}
}