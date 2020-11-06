#include "./TestSession.hpp"

#include "EnergyManager/Testing/Persistence/TestResults.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Persistence {
			void TestSession::onSave() {
				setID(insert("TestSession", { { "testName", '"' + filterSQL(getTestName()) + '"' }, { "profilerSessionID", Utility::Text::toString(getProfilingSession()->getID()) } }));

				getTestResults()->save();

				getProfilingSession()->save();
			}

			TestSession::TestSession(std::string testName, std::shared_ptr<TestResults> testResults, std::shared_ptr<Monitoring::Persistence::ProfilerSession> profilingSession)
				: testName_(std::move(testName))
				, testResults_(std::move(testResults))
				, profilingSession_(std::move(profilingSession)) {
				try {
					createTable("TestSession", { { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "testName", "TEXT NOT NULL" }, { "profilerSessionID", "INTEGER NOT NULL" } });
				} catch(const std::runtime_error& error) {
				}
			}

			std::string TestSession::getTestName() const {
				return testName_;
			}

			void TestSession::setTestName(const std::string& testName) {
				testName_ = testName;
			}

			std::shared_ptr<TestResults> TestSession::getTestResults() const {
				return testResults_;
			}

			void TestSession::setTestResults(const std::shared_ptr<TestResults>& testResults) {
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