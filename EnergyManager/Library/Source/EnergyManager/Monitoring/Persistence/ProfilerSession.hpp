#pragma once

#include "EnergyManager/Persistence/Entity.hpp"

#include <chrono>
#include <map>
#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		namespace Persistence {
			class MonitorData;

			/**
			 * Stores the results of a profiling operation.
			 */
			class ProfilerSession : public EnergyManager::Persistence::Entity {
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
				std::vector<std::shared_ptr<MonitorData>> monitorData_;

			protected:
				void onSave() final;

			public:
				/**
				 * Creates a new ProfilerSession.
				 * @param label The label of the session.
				 * @param profile The corresponding profile.
				 * @param monitorData The Monitor data.
				 */
				explicit ProfilerSession(std::string label, std::map<std::string, std::string> profile, std::vector<std::shared_ptr<MonitorData>> monitorData = {});

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
				std::vector<std::shared_ptr<MonitorData>> getMonitorData() const;

				/**
				 * Sets the Monitor data.
				 * @param monitorData The Monitor data.
				 */
				void setMonitorData(const std::vector<std::shared_ptr<MonitorData>>& monitorData);
			};
		}
	}
}