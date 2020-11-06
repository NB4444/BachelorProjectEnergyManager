#include "./TestRunner.hpp"

#include "EnergyManager/Testing/Persistence/TestResults.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		void TestRunner::onRun() {
			for(size_t testIndex = 0u; testIndex < tests_.size(); ++testIndex) {
				auto test = tests_[testIndex];

				// Execute the test and retrieve the results
				Utility::Logging::logInformation("Running test %s (%d/%d)...", test->getName().c_str(), testIndex + 1, tests_.size());
				try {
					test->run();

					//// Pretty-print the results
					//Utility::Logging::logInformation("Test completed with the following results:");
					//for(const auto& result : test->getTestResults()) {
					//	auto name = result.first;
					//	auto value = result.second;
					//
					//	Utility::Logging::logInformation("\t%s = %s", name.c_str(), value.c_str());
					//}
				} catch(const EnergyManager::Utility::Exceptions::Exception& exception) {
					exception.log();
				} catch(const std::exception& exception) {
					EnergyManager::Utility::Exceptions::Exception(exception.what(), __FILE__, __LINE__).log();
				}
			}
		}

		void TestRunner::afterRun() {
			testSessions_.clear();

			for(auto& test : getTests()) {
				Utility::Logging::logInformation("Storing test %s results...", test->getName().c_str());

				// Set up the session
				auto testSession = std::make_shared<Persistence::TestSession>(test->getName(), nullptr, test->getProfilerSessions()[0]);

				// Set up the results and add them to the session
				auto testResults = std::make_shared<Persistence::TestResults>(test->getTestResults(), testSession);
				testSession->setTestResults(testResults);

				testSessions_.push_back(testSession);
			}
		}

		TestRunner::TestRunner(std::vector<std::shared_ptr<Tests::Test>> tests) : tests_(std::move(tests)) {
		}

		std::vector<std::shared_ptr<Tests::Test>> TestRunner::getTests() const {
			return tests_;
		}

		void TestRunner::addTest(const std::shared_ptr<Tests::Test>& test) {
			tests_.push_back(test);
		}

		std::vector<std::shared_ptr<Persistence::TestSession>> TestRunner::getTestSessions() const {
			return testSessions_;
		}
	}
}