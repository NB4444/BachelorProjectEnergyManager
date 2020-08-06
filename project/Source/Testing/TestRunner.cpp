#include "./TestRunner.hpp"

#include "Testing/TestResults.hpp"

namespace Testing {
	void TestRunner::addTest(const Test& test) {
		tests_.push_back(test);
	}

	std::vector<TestResults> TestRunner::run() {
		std::vector<TestResults> results;

		for(auto& test : tests_) {
			results.push_back(test.run());
		}

		return results;
	}
}