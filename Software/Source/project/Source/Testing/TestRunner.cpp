#include "TestRunner.hpp"

namespace Testing {
	Testing::TestRunner::TestRunner(Testing::TestResultsStorage testResults)
		: testResults_(testResults) {
	}

	void TestRunner::addTest(const Test& test) {
		tests_.push_back(test);
	}

	void TestRunner::run() {
		for(auto& test : tests_) {
			testResults_.insert(test.run());
		}
	}
}