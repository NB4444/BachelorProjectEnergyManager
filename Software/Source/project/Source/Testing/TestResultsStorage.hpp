#pragma once

#include <filesystem>
#include <map>
#include <string>

namespace Testing {
	/**
	 * Responsible for managing Test results.
	 */
	class TestResultsStorage {
	public:
		/**
		 * Creates a new TestResultsStorage handler.
		 * @param databasePath The path to the sqlite database that contains the test results. Will be created if it does not exist.
		 */
		TestResultsStorage(const std::filesystem::path& databasePath);

		/**
		 * Inserts a new Test result.
		 */
		void insert(const std::map<std::string, std::string>& result);
	};
}