#include "./SyntheticGPUWorkloadTest.hpp"

#include "EnergyManager/Monitoring/CPUMonitor.hpp"
#include "EnergyManager/Monitoring/GPUMonitor.hpp"
#include "EnergyManager/Monitoring/NodeMonitor.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> SyntheticGPUWorkloadTest::onRun() {
				workload_->run();

				return {};
			}

			SyntheticGPUWorkloadTest::SyntheticGPUWorkloadTest(
				const std::string& name,
				std::shared_ptr<Benchmarking::Workloads::SyntheticGPUWorkload> workload,
				const std::shared_ptr<Hardware::Node>& node,
				const std::shared_ptr<Hardware::CPU>& cpu,
				const std::shared_ptr<Hardware::GPU>& gpu,
				std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors)
				: Test(name, monitors)
				, workload_(std::move(workload)) {
			}
		}
	}
}