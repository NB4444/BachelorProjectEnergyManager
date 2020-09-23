#pragma once

#include "EnergyManager/Benchmarking/Workloads/SyntheticGPUWorkload.hpp"
#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Hardware/Node.hpp"
#include "EnergyManager/Testing/Tests/ApplicationTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			class SyntheticGPUWorkloadTest : public Test {
				std::shared_ptr<Benchmarking::Workloads::SyntheticGPUWorkload> workload_;

			protected:
				std::map<std::string, std::string> onRun() override;

			public:
				SyntheticGPUWorkloadTest(
					const std::string& name,
					std::shared_ptr<Benchmarking::Workloads::SyntheticGPUWorkload> workload,
					const std::shared_ptr<Hardware::Node>& node,
					const std::shared_ptr<Hardware::CPU>& cpu,
					const std::shared_ptr<Hardware::GPU>& gpu,
					std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors = {});
			};
		}
	}
}