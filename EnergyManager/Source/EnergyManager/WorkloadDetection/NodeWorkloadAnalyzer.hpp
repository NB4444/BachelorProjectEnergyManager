#pragma once

#include "EnergyManager/Hardware/Node.hpp"
#include "EnergyManager/WorkloadDetection/WorkloadAnalyzer.hpp"

namespace EnergyManager {
	namespace WorkloadDetection {
		class NodeWorkloadAnalyzer : public WorkloadAnalyzer {
			std::shared_ptr<Hardware::Node> node_;

			// TODO: Initialize CPU and GPU workload analyzers as members

		protected:
			std::shared_ptr<Workloads::Workload> onAnalyzeWorkload() override;

		public:
			NodeWorkloadAnalyzer(std::shared_ptr<Hardware::Node> node);
		};
	}
}