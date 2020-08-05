#pragma once

#include "Testing/Test.hpp"

#include <vector>

namespace Testing {
	class TestResults;

	/**
	 * Used to run Tests and store their results.
	 */
	class TestRunner {
		/**
		 * The Tests to execute.
		 */
		std::vector<Test> tests_;

	public:
		/**
		 * Creates a new TestRunner.
		 */
		TestRunner() = default;

		/**
		 * Adds a Test to run.
		 * @param test The Test to add.
		 */
		void addTest(const Test& test);

		/**
		 * Runs the Tests.
		 */
		std::vector<TestResults> run();
	};
}