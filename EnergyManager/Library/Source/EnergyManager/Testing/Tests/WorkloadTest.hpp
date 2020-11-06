#pragma once

#include "EnergyManager/Benchmarking/Workloads/Workload.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/Tests/Test.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			/**
			 * Tests a Workload.
			 */
			class WorkloadTest : public Test {
				/**
				 * The Workload to test.
				 */
				std::shared_ptr<Benchmarking::Workloads::Workload> workload_;

			protected:
				std::map<std::string, std::string> onTest() override;

			public:
				/**
				 * Creates a new WorkloadTest.
				 * @param name The name of the Test.
				 * @param workload The Workload to run.
				 * @param monitors The Monitors to run during the Test.
				 */
				WorkloadTest(const std::string& name, std::shared_ptr<Benchmarking::Workloads::Workload> workload, const std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>>& monitors = {});
			};
		}
	}
}