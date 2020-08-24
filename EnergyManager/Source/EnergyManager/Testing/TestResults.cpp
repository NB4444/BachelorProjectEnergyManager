#include "./TestResults.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		void TestResults::onSave() {
			// Insert the Test results
			std::vector<std::map<std::string, std::string>> testRows;
			for(const auto& result : getResults()) {
				testRows.push_back({ { "test", "\"" + getTest().getName() + "\"" }, { "name", "\"" + result.first + "\"" }, { "value", "\"" + result.second + "\"" } });
			}
			insert("TestResults", testRows);

			// Insert the Monitor results
			std::vector<std::map<std::string, std::string>> monitorRows;
			for(const auto& monitorResult : getMonitorResults()) {
				std::string monitor = monitorResult.first->getName();

				for(const auto& timestampResults : monitorResult.second) {
					std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(timestampResults.first.time_since_epoch()).count());

					for(const auto& variableValue : timestampResults.second) {
						std::string name = variableValue.first;
						std::string value = variableValue.second;

						monitorRows.push_back({
							{ "test", "\"" + getTest().getName() + "\"" },
							{ "monitor", "\"" + monitor + "\"" },
							{ "timestamp", timestamp },
							{ "name", "\"" + name + "\"" },
							{ "value", "\"" + value + "\"" }
						});
					}
				}
			}
			insert("MonitorResults", monitorRows);
		}

		TestResults::TestResults(
			const std::string& databaseFile,
			Tests::Test test,
			std::map<std::string, std::string> results,
			std::map<std::shared_ptr<Profiling::Monitor>, std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>> monitorResults)
			: test_(std::move(test)), results_(std::move(results)), monitorResults_(std::move(monitorResults)) {
			// Create the tables if they do not exist yet
			try {
				createTable("TestResults", { { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "test", "TEXT NOT NULL" }, { "name", "TEXT NOT NULL" }, { "value", "TEXT" } });
			} catch(const std::runtime_error& error) {
			}
			try {
				createTable(
					"MonitorResults",
					{
						{ "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" },
						{ "test", "TEXT NOT NULL" },
						{ "monitor", "TEXT NOT NULL" },
						{ "timestamp", "INTEGER NOT NULL" },
						{ "name", "TEXT NOT NULL" },
						{ "value", "TEXT" }
					});
			} catch(const std::runtime_error& error) {
			}
		}

		Tests::Test TestResults::getTest() const {
			return test_;
		}

		std::map<std::string, std::string> TestResults::getResults() const {
			return results_;
		}

		std::map<std::shared_ptr<Profiling::Monitor>, std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>> TestResults::getMonitorResults() {
			return monitorResults_;
		}
	}
}