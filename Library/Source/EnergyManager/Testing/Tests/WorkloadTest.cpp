#include "./WorkloadTest.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> WorkloadTest::onTest() {
				workload_->run();

				return {};
			}

			WorkloadTest::WorkloadTest(
				const std::string& name,
				std::shared_ptr<Benchmarking::Workloads::Workload> workload,
				const std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>>& monitors)
				: Test(name, monitors)
				, workload_(std::move(workload)) {
			}
		}
	}
}