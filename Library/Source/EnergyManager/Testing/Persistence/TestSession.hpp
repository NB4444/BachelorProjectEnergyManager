#pragma once

#include "EnergyManager/Profiling/Persistence/ProfilerSession.hpp"
#include "EnergyManager/Utility/Persistence/Entity.hpp"

#include <chrono>
#include <map>
#include <memory>

namespace EnergyManager {
	namespace Testing {
		namespace Persistence {
			/**
			 * Stores the results of a profiling operation.
			 */
			class TestSession : public Utility::Persistence::Entity {
				/**
				 * The name of the Test that generated the results.
				 */
				std::string testName_;

				/**
				 * The Test results.
				 */
				std::map<std::string, std::string> testResults_;

				/**
				 * The profiling session that ran during the Test.
				 */
				std::shared_ptr<Profiling::Persistence::ProfilerSession> profilingSession_;

			protected:
				void onSave() final;

			public:
				/**
				 * Creates a new TestSession.
				 * @param testName The name of the Test that generated the results.
				 * @param testResults The Test results.
				 * @param profilingSession The profiling session that ran during the Test.
				 */
				explicit TestSession(std::string testName, std::map<std::string, std::string> testResults, std::shared_ptr<Profiling::Persistence::ProfilerSession> profilingSession = nullptr);

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
				 * Gets the results.
				 * @return The results.
				 */
				std::map<std::string, std::string> getTestResults() const;

				/**
				 * Sets the results.
				 * @param testResults The results.
				 */
				void setTestResults(const std::map<std::string, std::string>& testResults);

				/**
				 * Gets the profiling session that ran during the Test.
				 * @return The profiling session.
				 */
				std::shared_ptr<Profiling::Persistence::ProfilerSession> getProfilingSession() const;

				/**
				 * Sets the profiling session that ran during the Test.
				 * @param profilingSession The profiling session.
				 */
				void setProfilingSession(const std::shared_ptr<Profiling::Persistence::ProfilerSession>& profilingSession);
			};
		}
	}
}