#pragma once

#include "EnergyManager/WorkloadDetection/Workloads/Workload.hpp"

#include <memory>

namespace EnergyManager {
	namespace WorkloadDetection {
		class WorkloadAnalyzer {
		protected:
			virtual std::shared_ptr<Workloads::Workload> onAnalyzeWorkload() = 0;

		public:
			std::shared_ptr<Workloads::Workload> analyzeWorkload();
		};
	}
}