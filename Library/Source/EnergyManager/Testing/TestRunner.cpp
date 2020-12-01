#include "./TestRunner.hpp"

#include "EnergyManager/Testing/Persistence/TestSession.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		void TestRunner::onRun() {
			for(size_t testIndex = 0u; testIndex < tests_.size(); ++testIndex) {
				auto test = tests_[testIndex];

				// Execute the test and retrieve the results
				logInformation("Running test %s (%d/%d)...", test->getName().c_str(), testIndex + 1, tests_.size());
				try {
					test->run();
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
				logInformation("Storing test %s results...", test->getName().c_str());

				// Set up the session
				testSessions_.push_back(std::make_shared<Persistence::TestSession>(test->getName(), test->getTestResults(), test->getProfilerSessions()[0]));
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