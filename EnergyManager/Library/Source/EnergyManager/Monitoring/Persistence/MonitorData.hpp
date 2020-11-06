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
			class MonitorData : public EnergyManager::Persistence::Entity {
				/**
				 * The session that generated the data.
				 */
				std::shared_ptr<ProfilerSession> profilerSession_;

				/**
				 * The name of the Monitor that generated the data.
				 */
				std::string monitorName_;

				/**
				 * The data.
				 */
				std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> monitorData_;

			protected:
				void onSave() final;

			public:
				/**
				 * Creates a new MonitorData set.
				 * @param monitorData The data.
				 * @param monitorName The name of the Monitor that generated the data.
				 * @param profilerSession The session the data is associated with.
				 */
				MonitorData(
					std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> monitorData,
					std::string monitorName,
					std::shared_ptr<ProfilerSession> profilerSession = nullptr);

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