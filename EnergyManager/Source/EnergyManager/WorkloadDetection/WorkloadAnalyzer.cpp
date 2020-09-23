#include "./WorkloadAnalyzer.hpp"

namespace EnergyManager {
	namespace WorkloadDetection {
		std::shared_ptr<Workloads::Workload> WorkloadAnalyzer::analyzeWorkload() {
			onAnalyzeWorkload();
		}
	}
}