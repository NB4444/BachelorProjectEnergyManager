#pragma once

#include "EnergyManager/Monitoring/Persistence/ProfilerSession.hpp"
#include "EnergyManager/Persistence/Entity.hpp"

#include <chrono>
#include <map>
#include <memory>

namespace EnergyManager {
	namespace Testing {
		namespace Persistence {
			class TestResults;

			/**
			 * Stores the results of a profiling operation.
			 */
			class TestSession : public EnergyManager::Persistence::Entity {
				/**
				 * The name of the Test that generated the results.
				 */
				std::string testName_;

				/**
				 * The Test results.
				 */
				std::shared_ptr<TestResults> testResults_;

				/**
				 * The profiling session that ran during the Test.
				 */
				std::shared_ptr<Monitoring::Persistence::ProfilerSession> profilingSession_;

			protected:
				void onSave() final;

			public:
				/**
				 * Creates a new TestSession.
				 * @param testName The name of the Test that generated the results.
				 * @param testResults The Test results.
				 * @param profilingSession The profiling session that ran during the Test.
				 */
				explicit TestSession(std::string testName, std::shared_ptr<TestResults> testResults = nullptr, std::shared_ptr<Monitoring::Persistence::ProfilerSession> profilingSession = nullptr);

				/**
				 * Gets the name of the Test that generated the results.
				 * @return The Test name.
				 */
				std::string getTestName() const;

				/**
				 * Sets the name of the Test that generated the results.
				 * @param testName The Test name.
				 */
				void setTestName(const std::string& testName);

				/**
				 * Gets the Test results.
				 * @return The Test results.
				 */
				std::shared_ptr<TestResults> getTestResults() const;

				/**
				 * Sets the Test results.
				 * @param testResults The Test results.
				 */
				void setTestResults(const std::shared_ptr<TestResults>& testResults);

				/**
				 * Gets the profiling session that ran during the Test.
				 * @return The profiling session.
				 */
				std::shared_ptr<Monitoring::Persistence::ProfilerSession> getProfilingSession() const;

				/**
				 * Sets the profiling session that ran during the Test.
				 * @param profilingSession The profiling session.
				 */
				void setProfilingSession(const std::shared_ptr<Monitoring::Persistence::ProfilerSession>& profilingSession);
			};
		}
	}
}