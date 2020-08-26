#pragma once

#include "EnergyManager/Testing/Tests/Test.hpp"
#include "EnergyManager/Testing/TestResults.hpp"

#include <memory>
#include <vector>

namespace EnergyManager {
	namespace Testing {
		/**
		 * Used to run Tests and store their results.
		 */
		class TestRunner {
				/**
				 * The Tests to execute.
				 */
				std::vector<std::shared_ptr<Tests::Test>> tests_;

			public:
				/**
				 * Creates a new TestRunner.
				 */
				TestRunner() = default;

				/**
				 * Adds a Test to run.
				 * @param test The Test to add.
				 */
				void addTest(const std::shared_ptr<Tests::Test>& test);

				/**
				 * Runs the Tests.
				 * @param databaseFile The database file to use.
				 */
				std::vector<TestResults> run(const std::string& databaseFile);
		};
	}
}