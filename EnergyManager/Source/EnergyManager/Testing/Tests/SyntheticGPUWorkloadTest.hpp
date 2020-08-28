#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Hardware/Node.hpp"
#include "EnergyManager/Testing/Benchmarking/SyntheticGPUWorkload.hpp"
#include "EnergyManager/Testing/Tests/ApplicationTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			class SyntheticGPUWorkloadTest : public Test {
				std::shared_ptr<Benchmarking::SyntheticGPUWorkload> workload_;

			protected:
				std::map<std::string, std::string> onRun() override;

			public:
				SyntheticGPUWorkloadTest(
					const std::string& name,
					std::shared_ptr<Benchmarking::SyntheticGPUWorkload> workload,
					const std::shared_ptr<Hardware::Node>& node,
					const std::shared_ptr<Hardware::CPU>& cpu,
					const std::shared_ptr<Hardware::GPU>& gpu);
			};
		}
	}
}