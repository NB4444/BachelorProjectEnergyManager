#pragma once

#include "EnergyManager/Persistence/Entity.hpp"
#include "EnergyManager/Testing/Tests/Test.hpp"

#include <map>
#include <string>
#include <chrono>

namespace EnergyManager {
	namespace Testing {
		/**
		 * Represents the results of a single Test.
		 */
		class TestResults : public Persistence::Entity<TestResults> {
			/**
			 * The Test that generated the results.
			 */
			Tests::Test test_;

			/**
			 * The actual result values.
			 */
			std::map<std::string, std::string> results_;

			/**
			 * The results of the Monitors.
			 */
			std::map<std::shared_ptr<Profiling::Monitor>, std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>> monitorResults_;

			void onSave() override;

		public:
			/**
			 * Creates a new TestResults set.
			 * @param test The Test that generated the results.
			 * @param results The actual result values.
			 * @param monitorResults The results of the Monitors.
			 */
			TestResults(Tests::Test test, std::map<std::string, std::string> results = {}, std::map<std::shared_ptr<Profiling::Monitor>, std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>> monitorResults = {});

			/**
			 * Gets the Test that generated the results.
			 * @return The Test.
			 */
			Tests::Test getTest() const;

			/**
			 * Gets the actual result values.
			 * @return The result values.
			 */
			std::map<std::string, std::string> getResults() const;

			/**
			 * Gets the results of the Monitors.
			 * @return the Monitor results.
			 */
			std::map<std::shared_ptr<Profiling::Monitor>, std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>> getMonitorResults();
		};
	}
}