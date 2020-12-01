#pragma once

#include "EnergyManager/Testing/Tests/ApplicationTest.hpp"

#include <string>
#include <map>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			/**
			 * Tests the ping function.
			 */
			class PingTest : public EnergyManager::Testing::Tests::ApplicationTest {
			public:
				/**
				 * Creates a new PingTest.
				 * @param name The name of the Test.
				 * @param host The host to ping.
				 * @param times The amount of times to ping.
				 * @param applicationMonitorPollingInterval The interval used to monitor the application.
				 * @param monitors The Monitors to use.
				 */
				PingTest(
					const std::string& name,
					const std::string& host,
					const unsigned int& times,
					const std::chrono::system_clock::duration& applicationMonitorPollingInterval,
					const std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>>& monitors = {});

				/**
				 * Creates a new PingTest from command line arguments.
				 * @param arguments The command line arguments.
				 */
				explicit PingTest(const std::map<std::string, std::string>& arguments);
			};
		}
	}
}