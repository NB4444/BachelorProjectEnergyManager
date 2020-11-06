#include "./ApplicationTest.hpp"

#include "EnergyManager/Monitoring/Monitors/ApplicationMonitor.hpp"
#include "EnergyManager/Testing/Application.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <regex>
#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> ApplicationTest::onTest() {
				// Keep track of Test results
				std::map<std::string, std::string> results;

				// Run the Application
				Utility::Logging::logInformation("Running application...");
				application_.run();

				// Get output
				std::string output = application_.getExecutableOutput();

				// Parse and add results
				for(const auto& result : results_) {
					std::smatch match;
					std::regex regex(result.second);
					if(std::regex_search(output, match, regex)) {
						results[result.first] = match.str(1);
					}
				}

				return results;
			}

			ApplicationTest::ApplicationTest(
				const std::string& name,
				const Application& application,
				std::map<std::string, std::string> results,
				const std::chrono::system_clock::duration& applicationMonitorPollingInterval,
				std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors)
				: application_(application)
				, Test(
					  name,
					  [&]() {
						  monitors.push_back(std::make_shared<Monitoring::Monitors::ApplicationMonitor>(application_, applicationMonitorPollingInterval));
						  return monitors;
					  }())
				, results_(std::move(results)) {
			}
		}
	}
}