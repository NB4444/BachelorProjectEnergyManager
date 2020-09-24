#pragma once

#include "ApplicationTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			class PingTest : public ApplicationTest {
			public:
				static void initialize();

				PingTest(
					const std::string& name,
					const std::string& host,
					const unsigned int& times,
					std::chrono::system_clock::duration applicationMonitorPollingInterval,
					std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors = {});
			};
		}
	}
}