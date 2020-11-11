#pragma once

#include "EnergyManager/Persistence/Entity.hpp"

#include <chrono>
#include <map>
#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		namespace Persistence {
			class ProfilerSession;

			/**
			 * Stores the results of a profiling operation.
			 */
			class MonitorSession : public EnergyManager::Persistence::Entity {
				/**
				 * The name of the Monitor that generated the data.
				 */
				std::string monitorName_;

				/**
				 * The Monitor data.
				 */
				std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> monitorData_;

				/**
				 * The session that generated the data.
				 */
				std::shared_ptr<ProfilerSession> profilerSession_;

			protected:
				void onSave() final;

			public:
				/**
				 * Creates a new MonitorSession.
				 * @param monitorName The name of the Monitor that generated the data.
				 * @param monitorData The Monitor data.
				 * @param profilerSession The session the data is associated with.
				 */
				explicit MonitorSession(
					std::string monitorName,
					std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> monitorData,
					std::shared_ptr<ProfilerSession> profilerSession = nullptr);

				/**
				 * Gets the name of the Monitor that generated the data.
				 * @return The Monitor name.
				 */
				std::string getMonitorName() const;

				/**
				 * Sets the name of the Monitor that generated the data.
				 * @param monitorName The Monitor name.
				 */
				void setMonitorName(const std::string& monitorName);

				/**
				 * Gets the data.
				 * @return The data.
				 */
				std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> getMonitorData() const;

				/**
				 * Sets the data.
				 * @param monitorData The data.
				 */
				void setMonitorData(const std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>& monitorData);

				/**
				 * Gets the session that generated the data.
				 * @return The session.
				 */
				std::shared_ptr<ProfilerSession> getProfilerSession() const;

				/**
				 * Sets the session that generated the data.
				 * @param profilerSession The session.
				 */
				void setProfilerSession(const std::shared_ptr<ProfilerSession>& profilerSession);
			};
		}
	}
}