#include "./TestResults.hpp"

#include "EnergyManager/Testing/Persistence/TestSession.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Persistence {
			void TestResults::onSave() {
				std::vector<std::map<std::string, std::string>> testResults;
				for(const auto& result : getTestResults()) {
					testResults.push_back({ { "testSessionID", Utility::Text::toString(getTestSession()->getID()) },
											{ "name", '"' + filterSQL(result.first) + '"' },
											{ "value", '"' + filterSQL(result.second) + '"' } });
				}

				insert("TestResults", testResults);
			}

			TestResults::TestResults(std::map<std::string, std::string> testResults, std::shared_ptr<TestSession> testSession)
				: testResults_(std::move(testResults))
				, testSession_(std::move(testSession)) {
				try {
					createTable("TestResults", { { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "testSessionID", "INTEGER NOT NULL" }, { "name", "TEXT NOT NULL" }, { "value", "TEXT" } });
				} catch(const std::runtime_error& error) {
				}
			}

			std::map<std::string, std::string> TestResults::getTestResults() const {
				return testResults_;
			}

			void TestResults::setTestResults(const std::map<std::string, std::string>& testResults) {
				testResults_ = testResults;
			}

			std::shared_ptr<TestSession> TestResults::getTestSession() const {
				return testSession_;
			}

			void TestResults::setTestSession(const std::shared_ptr<TestSession>& testSession) {
				testSession_ = testSession;
			}
		}
	}
}