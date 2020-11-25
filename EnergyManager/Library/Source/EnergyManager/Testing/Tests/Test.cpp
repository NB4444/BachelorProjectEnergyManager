#include "./Test.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <mutex>
#include <regex>
#include <thread>
#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::vector<std::string> Test::generateHeaders() const {
				auto headers = Runnable::generateHeaders();
				headers.push_back("Test " + getName());

				return headers;
			}

			void Test::onProfile(const std::map<std::string, std::string>& profile) {
				// Run the Test
				logInformation("Running test...");
				testResults_ = onTest();
			}

			std::map<std::string, std::string> Test::onTest() {
				return {};
			}

			Test::Test(std::string name, const std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>>& monitors)
				: Monitoring::Profilers::Profiler(name, { { { "", "" } } }, monitors)
				, name_(std::move(name)) {
			}

			std::string Test::getName() const {
				return name_;
			}

			std::map<std::string, std::string> Test::getTestResults() const {
				return testResults_;
			}
		}
	}
}