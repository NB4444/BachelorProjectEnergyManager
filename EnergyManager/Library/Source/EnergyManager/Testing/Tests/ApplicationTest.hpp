#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/Tests/Test.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			/**
			 * Tests an Application's execution.
			 */
			class ApplicationTest : public Test {
				/**
				 * The Application to test.
				 */
				Application application_;

				/**
				 * The results to parse from the Application's output.
				 */
				std::map<std::string, std::string> results_;

			protected:
				std::map<std::string, std::string> onTest() final;

			public:
				/**
				 * Creates a new ApplicationTest.
				 * @param name The name of the ApplicationTest.
				 * @param application The Application to test.
				 * @param parameters The parameters to use to run the Application.
				 * @param results The results to parse from the Application's output.
				 * @param monitors The monitors to run during the Test and their associated polling intervals.
				 */
				ApplicationTest(
					const std::string& name,
					const Application& application,
					std::map<std::string, std::string> results,
					const std::chrono::system_clock::duration& applicationMonitorPollingInterval,
					std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors = {});
			};
		}
	}
}