#include "./TestSession.hpp"

#include "EnergyManager/Utility/Text.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Persistence {
			void TestSession::onSave() {
				setID(insert("TestSession", { { "testName", '\'' + filterSQL(getTestName()) + '\'' }, { "profilerSessionID", Utility::Text::toString(getProfilingSession()->getID()) } }));

				std::vector<std::map<std::string, std::string>> testResults;
				for(const auto& result : getTestResults()) {
					testResults.push_back(
						{ { "testSessionID", Utility::Text::toString(getID()) }, { "name", '\'' + filterSQL(result.first) + '\'' }, { "value", '\'' + filterSQL(result.second) + '\'' } });
				}
				insert("TestResults", testResults);

				getProfilingSession()->save();
			}

			TestSession::TestSession(std::string testName, std::map<std::string, std::string> testResults, std::shared_ptr<Monitoring::Persistence::ProfilerSession> profilingSession)
				: testName_(std::move(testName))
				, testResults_(std::move(testResults))
				, profilingSession_(std::move(profilingSession)) {
				try {
					createTable("TestSession", { { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "testName", "TEXT NOT NULL" }, { "profilerSessionID", "INTEGER NOT NULL" } });
				} catch(const std::runtime_error& error) {
				}
				try {
					createTable("TestResults", { { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "testSessionID", "INTEGER NOT NULL" }, { "name", "TEXT NOT NULL" }, { "value", "TEXT" } });
					createIndex("TestResults", "testSessionIDIndex", { "testSessionID" });
				} catch(const std::runtime_error& error) {
				}
			}

			std::string TestSession::getTestName() const {
				return testName_;
			}

			void TestSession::setTestName(const std::string& testName) {
				testName_ = testName;
			}

			std::map<std::string, std::string> TestSession::getTestResults() const {
				return testResults_;
			}

			void TestSession::setTestResults(const std::map<std::string, std::string>& testResults) {
				testResults_ = testResults;
			}

			std::shared_ptr<Monitoring::Persistence::ProfilerSession> TestSession::getProfilingSession() const {
				return profilingSession_;
			}

			void TestSession::setProfilingSession(const std::shared_ptr<Monitoring::Persistence::ProfilerSession>& profilingSession) {
				profilingSession_ = profilingSession;
			}
		}
	}
}