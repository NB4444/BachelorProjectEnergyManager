#pragma once

#include "EnergyManager/Utility/Persistence/Entity.hpp"

#include <chrono>
#include <map>
#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		namespace Persistence {
			class MonitorSession;
		}
	}

	namespace Profiling {
		namespace Persistence {
			/**
			 * Stores the results of a profiling operation.
			 */
			class ProfilerSession : public Utility::Persistence::Entity {
				/**
				 * The label of the session.
				 */
				std::string label_;

				/**
				 * The corresponding profile.
				 */
				std::map<std::string, std::string> profile_;

				/**
				 * The Monitor data.
				 */
				std::vector<std::shared_ptr<Monitoring::Persistence::MonitorSession>> monitorSessions_;

			protected:
				void onSave() final;

			public:
				/**
				 * Creates a new ProfilerSession.
				 * @param label The label of the session.
				 * @param profile The corresponding profile.
				 * @param monitorSessions The Monitor data.
				 */
				explicit ProfilerSession(std::string label, std::map<std::string, std::string> profile, std::vector<std::shared_ptr<Monitoring::Persistence::MonitorSession>> monitorSessions = {});

				/**
				 * Gets the label of the session.
				 * @return The label.
				 */
				std::string getLabel() const;

				/**
				 * Sets the label of the session.
				 * @param label The label.
				 */
				void setLabel(const std::string& label);

				/**
				 * Gets the corresponding profile.
				 * @return The profile.
				 */
				std::map<std::string, std::string> getProfile() const;

				/**
				 * Sets the corresponding profile.
				 * @param profile The profile.
				 */
				void setProfile(const std::map<std::string, std::string>& profile);

				/**
				 * Gets the Monitor data.
				 * @return The Monitor data.
				 */
				std::vector<std::shared_ptr<Monitoring::Persistence::MonitorSession>> getMonitorSessions() const;

				/**
				 * Sets the Monitor data.
				 * @param monitorSessions The Monitor data.
				 */
				void setMonitorSessions(const std::vector<std::shared_ptr<Monitoring::Persistence::MonitorSession>>& monitorSessions);
			};
		}
	}
}