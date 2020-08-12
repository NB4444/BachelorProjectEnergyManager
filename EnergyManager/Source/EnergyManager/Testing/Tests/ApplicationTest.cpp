#include "./ApplicationTest.hpp"

#include "EnergyManager/Application.hpp"
#include "EnergyManager/Testing/TestResults.hpp"

#include <regex>
#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> ApplicationTest::onRun() {
				// Keep track of Test results
				std::map<std::string, std::string> results;

				// Run the Application
				application_.run(parameters_);

				// Get output
				std::string output = application_.getExecutableOutput();

				// Add Application output
				results["output"] = output;

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

			ApplicationTest::ApplicationTest(const std::string& name, const Application& application, std::vector<std::string> parameters, std::map<std::string, std::string> results, std::map<std::shared_ptr<Profiling::Monitor>, std::chrono::seconds> monitors)
				: Test(name, monitors)
				, application_(application)
				, parameters_(std::move(parameters))
				, results_(std::move(results)) {
			}
		}
	}
}