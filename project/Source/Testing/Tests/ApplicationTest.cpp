#include "./ApplicationTest.hpp"

#include "Application.hpp"
#include "Testing/TestResults.hpp"

#include <regex>
#include <utility>

namespace Testing {
	namespace Tests {
		TestResults ApplicationTest::onRun() {
			// Keep track of Test results
			std::map<std::string, std::string> results;

			// Run the Application
			application_.start(parameters_);
			application_.waitUntilDone();

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

			return TestResults(*this, results);
		}

		ApplicationTest::ApplicationTest(const std::string& name, const Application& application, std::vector<std::string> parameters, std::map<std::string, std::string> results)
			: Test(name)
			, application_(application)
			, parameters_(std::move(parameters))
			, results_(std::move(results)) {
		}
	}
}