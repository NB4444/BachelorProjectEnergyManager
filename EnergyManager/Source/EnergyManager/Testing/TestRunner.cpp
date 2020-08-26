#include "./TestRunner.hpp"

#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Testing {
		void TestRunner::addTest(const std::shared_ptr<Tests::Test>& test) {
			tests_.push_back(test);
		}

		std::vector<TestResults> TestRunner::run(const std::string& databaseFile) {
			std::vector<TestResults> results;

			for(size_t testIndex = 0u; testIndex < tests_.size(); ++testIndex) {
				auto test = tests_[testIndex];
				auto name = test->getName();

				Utility::Logging::logInformation("Running test %s (%d/%d)...", name.c_str(), testIndex + 1, tests_.size());

				// Execute the test and retrieve the results
				TestResults testResults = test->run(databaseFile);

				Utility::Logging::logInformation("Test complete");

				// Pretty-print the results
				Utility::Logging::logInformation("Test results:");
				for(const auto& result : testResults.getResults()) {
					auto name = result.first;
					auto value = result.second;

					Utility::Logging::logInformation("\t%s = %s", name.c_str(), value.c_str());
				}

				// Pretty-print the results
				for(const auto& monitorResult : testResults.getMonitorResults()) {
					auto monitor = monitorResult.first;
					auto name = monitor->getName();

					Utility::Logging::logInformation("Monitor %s results:", name.c_str());

					auto timestampedValues = monitorResult.second;
					for(const auto& timestampedValue : timestampedValues) {
						auto timestamp = timestampedValue.first;
						auto variables = timestampedValue.second;

						for(const auto& variableValues : variables) {
							auto name = variableValues.first;
							auto value = variableValues.second;
							auto timestampString = Utility::Text::formatTimestamp(timestamp);

							Utility::Logging::logInformation("\t[%s] %s = %s", timestampString.c_str(), name.c_str(), value.c_str());
						}
					}
				}

				// Store the results to return them
				results.push_back(testResults);

				// Save the Test and results to the database
				testResults.save();
			}

			return results;
		}
	}
}