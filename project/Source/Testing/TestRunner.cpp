#include "./TestRunner.hpp"

#include "Testing/TestResults.hpp"

namespace Testing {
	void TestRunner::addTest(const std::shared_ptr<Tests::Test>& test) {
		tests_.push_back(test);
	}

	std::vector<TestResults> TestRunner::run() {
		std::vector<TestResults> results;

		for(auto& test : tests_) {
			// Execute the test and retrieve the results
			TestResults testResults = test->run();

			// Store the results to return them
			results.push_back(testResults);

			// Save the Test and results to the database
			test->save();
			testResults.save();
		}

		return results;
	}
}