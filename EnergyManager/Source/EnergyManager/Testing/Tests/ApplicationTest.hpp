#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/Tests/Test.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			class ApplicationTest : public Test {
				/**
				 * The Application to test.
				 */
				Application application_;

				/**
				 * The parameters to use to run the Application.
				 */
				std::vector<std::string> parameters_;

				/**
				 * The CPUs to run on.
				 */
				std::vector<std::shared_ptr<Hardware::CPU>> cpus_;

				/**
				 * The GPU to run on.
				 */
				std::shared_ptr<Hardware::GPU> gpu_;

				/**
				 * The results to parse from the Application's output.
				 */
				std::map<std::string, std::string> results_;

			protected:
				std::map<std::string, std::string> onRun() override;

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
					std::vector<std::string> parameters,
					std::vector<std::shared_ptr<Hardware::CPU>> cpus,
					std::shared_ptr<Hardware::GPU> gpu,
					std::map<std::string, std::string> results,
					std::chrono::system_clock::duration applicationMonitorPollingInterval,
					std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors = {});
			};
		}
	}
}