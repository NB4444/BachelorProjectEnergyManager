#pragma once

#include "EnergyManager/Persistence/Entity.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Persistence {
			class TestSession;

			/**
			 * Represents the results of a single Test.
			 */
			class TestResults : public EnergyManager::Persistence::Entity {
				/**
				 * The session that generated the results.
				 */
				std::shared_ptr<TestSession> testSession_;

				/**
				 * The results.
				 */
				std::map<std::string, std::string> testResults_;

			protected:
				void onSave() final;

			public:
				/**
				 * Creates a new TestResults set.
				 * @param testResults The results.
				 * @param testSession The session that generated the results.
				 */
				explicit TestResults(std::map<std::string, std::string> testResults, std::shared_ptr<TestSession> testSession = nullptr);

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
				 * Gets the session that generated the results.
				 * @return The Test session.
				 */
				std::shared_ptr<TestSession> getTestSession() const;

				/**
				 * Sets the session that generated the results.
				 * @param testSession The Test session.
				 */
				void setTestSession(const std::shared_ptr<TestSession>& testSession);
			};
		}
	}
}