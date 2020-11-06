#pragma once

#include "EnergyManager/Testing/Persistence/TestSession.hpp"
#include "EnergyManager/Testing/Tests/Test.hpp"
#include "EnergyManager/Utility/Runnable.hpp"

#include <memory>
#include <vector>

namespace EnergyManager {
	namespace Testing {
		/**
		 * Used to run Tests and store their results.
		 */
		class TestRunner : public Utility::Runnable {
			/**
			 * The Tests to execute.
			 */
			std::vector<std::shared_ptr<Tests::Test>> tests_;

			/**
			 * The results of the last test sessions.
			 */
			std::vector<std::shared_ptr<Persistence::TestSession>> testSessions_;

		protected:
			void onRun() final;

			void afterRun() final;

		public:
			/**
			 * Creates a new TestRunner.
			 */
			explicit TestRunner(std::vector<std::shared_ptr<Tests::Test>> tests);

			/**
			 * Gets the Tests to execute.
			 * @return The Tests.
			 */
			std::vector<std::shared_ptr<Tests::Test>> getTests() const;

			/**
			 * Adds a Test to run.
			 * @param test The Test to add.
			 */
			void addTest(const std::shared_ptr<Tests::Test>& test);

			/**
			 * Gets the results of the last test session.
			 * @return The test session.
			 */
			std::vector<std::shared_ptr<Persistence::TestSession>> getTestSessions() const;
		};
	}
}