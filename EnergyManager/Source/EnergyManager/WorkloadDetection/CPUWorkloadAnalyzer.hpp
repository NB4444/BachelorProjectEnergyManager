#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/WorkloadDetection/WorkloadAnalyzer.hpp"

namespace EnergyManager {
	namespace WorkloadDetection {
		class CPUWorkloadAnalyzer : public WorkloadAnalyzer {
			std::shared_ptr<Hardware::CPU> cpu_;

		protected:
			std::shared_ptr<Workloads::Workload> onAnalyzeWorkload() override;

		public:
			CPUWorkloadAnalyzer(std::shared_ptr<Hardware::CPU> cpu);
		};
	}
}