#pragma once

#include "EnergyManager/Testing/Tests/ApplicationTest.hpp"

#include <chrono>
#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			class MatrixMultiplyTest : public ApplicationTest {
			public:
				static void initialize();

				MatrixMultiplyTest(
					const std::string& name,
					const std::vector<std::shared_ptr<Hardware::CPU>>& cpus,
					const std::shared_ptr<Hardware::GPU>& gpu,
					const size_t& matrixAWidth,
					const size_t& matrixAHeight,
					const size_t& matrixBWidth,
					const size_t& matrixBHeight,
					std::chrono::system_clock::duration applicationMonitorPollingInterval,
					const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors = {});
			};
		}
	}
}