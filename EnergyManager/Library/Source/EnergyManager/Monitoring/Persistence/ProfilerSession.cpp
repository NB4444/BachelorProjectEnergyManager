#include "./ProfilerSession.hpp"

#include "EnergyManager/Monitoring/Persistence/MonitorSession.hpp"

#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Persistence {
			void ProfilerSession::onSave() {
				setID(insert("ProfilerSession", { { "label", '"' + getLabel() + '"' } }));

				std::vector<std::map<std::string, std::string>> profileRows;
				for(const auto& argument : getProfile()) {
					profileRows.push_back(
						{ { "profilerSessionID", Utility::Text::toString(getID()) }, { "argument", '"' + filterSQL(argument.first) + '"' }, { "value", '"' + filterSQL(argument.second) + '"' } });
				}
				insert("ProfilerSessionProfile", profileRows);

				for(const auto& monitorSession : getMonitorSessions()) {
					monitorSession->save();
				}
			}

			ProfilerSession::ProfilerSession(std::string label, std::map<std::string, std::string> profile, std::vector<std::shared_ptr<MonitorSession>> monitorSessions)
				: label_(std::move(label))
				, profile_(std::move(profile))
				, monitorSessions_(std::move(monitorSessions)) {
				try {
					createTable("ProfilerSession", { { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "label", "TEXT NOT NULL" } });
				} catch(const std::runtime_error& error) {
				}
				try {
					createTable(
						"ProfilerSessionProfile",
						{ { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "profilerSessionID", "INTEGER NOT NULL" }, { "argument", "TEXT NOT NULL" }, { "value", "TEXT NOT NULL" } });
					createIndex("ProfilerSessionProfile", "profilerSessionIDIndex", { "profilerSessionID" });
				} catch(const std::runtime_error& error) {
				}
			}

			std::string ProfilerSession::getLabel() const {
				return label_;
			}

			void ProfilerSession::setLabel(const std::string& label) {
				label_ = label;
			}

			std::map<std::string, std::string> ProfilerSession::getProfile() const {
				return profile_;
			}

			void ProfilerSession::setProfile(const std::map<std::string, std::string>& profile) {
				profile_ = profile;
			}

			std::vector<std::shared_ptr<MonitorSession>> ProfilerSession::getMonitorSessions() const {
				return monitorSessions_;
			}

			void ProfilerSession::setMonitorSessions(const std::vector<std::shared_ptr<MonitorSession>>& monitorSessions) {
				monitorSessions_ = monitorSessions;
			}
		}
	}
}