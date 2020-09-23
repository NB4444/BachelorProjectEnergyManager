#pragma once

#include "EnergyManager/Hardware/Node.hpp"
#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/Tests/ApplicationTest.hpp"

#include <string>
#include <chrono>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			class MatrixMultiplyTest : public ApplicationTest {
			public:
				MatrixMultiplyTest(
					const std::string& name,
					const std::shared_ptr<Hardware::Node>& node,
					const std::shared_ptr<Hardware::CPU>& cpu,
					const std::shared_ptr<Hardware::GPU>& gpu,
					const size_t& matrixAWidth,
					const size_t& matrixAHeight,
					const size_t& matrixBWidth,
					const size_t& matrixBHeight,
					std::chrono::system_clock::duration applicationMonitorPollingInterval,
					std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors = {});
			};
		}
	}
}