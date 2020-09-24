#pragma once

#include "EnergyManager/Benchmarking/Workloads/SyntheticWorkload.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/Tests/Test.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			class SyntheticWorkloadTest : public Test {
				std::shared_ptr<Benchmarking::Workloads::SyntheticWorkload> workload_;

				/**
				 * The GPU to run on.
				 */
				std::shared_ptr<Hardware::GPU> gpu_;

			protected:
				std::map<std::string, std::string> onRun() override;

			public:
				SyntheticWorkloadTest(
					const std::string& name,
					std::shared_ptr<Benchmarking::Workloads::SyntheticWorkload> workload,
					std::shared_ptr<Hardware::GPU> gpu,
					std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors = {});
			};
		}
	}
}