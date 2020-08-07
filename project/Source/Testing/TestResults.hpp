#pragma once

#include "Persistence/Entity.hpp"
#include "Testing/Tests/Test.hpp"

#include <map>
#include <string>

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

		std::map<std::string, std::string> onSave() override;

	public:
		TestResults(const std::map<std::string, std::string>& row);

		/**
		 * Creates a new TestResults set.
		 * @param test The Test that generated the results.
		 * @param results The actual result values.
		 */
		TestResults(Tests::Test test, std::map<std::string, std::string> results);

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
	};
}