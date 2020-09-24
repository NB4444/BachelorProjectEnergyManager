#include "./SyntheticWorkloadTest.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> SyntheticWorkloadTest::onRun() {
				workload_->run(gpu_);

				return {};
			}

			SyntheticWorkloadTest::SyntheticWorkloadTest(
				const std::string& name,
				std::shared_ptr<Benchmarking::Workloads::SyntheticWorkload> workload,
				std::shared_ptr<Hardware::GPU> gpu,
				std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors)
				: Test(name, std::move(monitors))
				, workload_(std::move(workload))
				, gpu_(std::move(gpu)) {
			}
		}
	}
}