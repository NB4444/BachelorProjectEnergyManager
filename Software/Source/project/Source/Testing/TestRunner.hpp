#pragma once

#include "Testing/Test.hpp"
#include "Testing/TestResultsStorage.hpp"

namespace Testing {
	/**
	 * Used to run Tests and store their results.
	 */
	class TestRunner {
		/**
		 * The database containing Test results.
		 */
		TestResultsStorage testResults_;

		/**
		 * The Tests to execute.
		 */
		std::vector<Test> tests_;

	public:
		/**
		 * Creates a new TestRunner.
		 * @param testResults The database containing Test results.
		 */
		TestRunner(TestResultsStorage testResults);

		/**
		 * Adds a Test to run.
		 * @param test The Test to add.
		 */
		void addTest(const Test& test);

		/**
		 * Runs the Tests.
		 */
		void run();
	};
}