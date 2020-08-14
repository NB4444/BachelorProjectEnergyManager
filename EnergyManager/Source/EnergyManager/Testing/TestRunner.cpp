#include "./TestRunner.hpp"

#include "TestResults.hpp"

namespace EnergyManager {
	namespace Testing {
		void TestRunner::addTest(const std::shared_ptr<Tests::Test>& test) {
			tests_.push_back(test);
		}

		std::vector<TestResults> TestRunner::run(const std::string& databaseFile) {
			std::vector<TestResults> results;

			for(auto& test : tests_) {
				// Execute the test and retrieve the results
				TestResults testResults = test->run(databaseFile);

				// Store the results to return them
				results.push_back(testResults);

				// Save the Test and results to the database
				testResults.save();
			}

			return results;
		}
	}
}