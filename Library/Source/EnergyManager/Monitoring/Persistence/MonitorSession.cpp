#include "./MonitorSession.hpp"

#include "EnergyManager/Profiling/Persistence/ProfilerSession.hpp"

#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Persistence {
			void MonitorSession::onSave() {
				setID(insert("MonitorSession", { { "profilerSessionID", Utility::Text::toString(getProfilerSession()->getID()) }, { "monitorName", '\'' + filterSQL(getMonitorName()) + '\'' } }));

				std::vector<std::map<std::string, std::string>> monitorData;
				for(const auto& currentMonitorData : getMonitorData()) {
					std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(currentMonitorData.first.time_since_epoch()).count());

					for(const auto& variableValue : currentMonitorData.second) {
						std::string name = variableValue.first;
						std::string value = variableValue.second;

						// Filter the value
						const std::string find = "\"";
						size_t startIndex = value.find(find);
						if(startIndex != std::string::npos) {
							value.replace(startIndex, find.length(), "\"\"");
						}

						monitorData.push_back(
							{ { "monitorSessionID", std::to_string(getID()) }, { "timestamp", timestamp }, { "name", '\'' + filterSQL(name) + '\'' }, { "value", '\'' + filterSQL(value) + '\'' } });
					}
				}
				insert("MonitorData", monitorData);
			}

			MonitorSession::MonitorSession(
				std::string monitorName,
				std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> monitorData,
				std::shared_ptr<Profiling::Persistence::ProfilerSession> profilerSession)
				: monitorName_(std::move(monitorName))
				, monitorData_(std::move(monitorData))
				, profilerSession_(std::move(profilerSession)) {
				try {
					createTable("MonitorSession", { { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "profilerSessionID", "INTEGER NOT NULL" }, { "monitorName", "TEXT NOT NULL" } });
					createIndex("MonitorSession", "profilerSessionIDIndex", { "profilerSessionID" });
				} catch(const std::runtime_error& error) {
				}
				try {
					createTable(
						"MonitorData",
						{ { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" },
						  { "monitorSessionID", "INTEGER NOT NULL" },
						  { "timestamp", "INTEGER NOT NULL" },
						  { "name", "TEXT NOT NULL" },
						  { "value", "TEXT" } });
					createIndex("MonitorData", "monitorSessionIDIndex", { "monitorSessionID" });
				} catch(const std::runtime_error& error) {
				}
			}

			std::string MonitorSession::getMonitorName() const {
				return monitorName_;
			}

			void MonitorSession::setMonitorName(const std::string& monitorName) {
				monitorName_ = monitorName;
			}

			std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> MonitorSession::getMonitorData() const {
				return monitorData_;
			}

			void MonitorSession::setMonitorData(const std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>& monitorData) {
				monitorData_ = monitorData;
			}

			std::shared_ptr<Profiling::Persistence::ProfilerSession> MonitorSession::getProfilerSession() const {
				return profilerSession_;
			}

			void MonitorSession::setProfilerSession(const std::shared_ptr<Profiling::Persistence::ProfilerSession>& profilerSession) {
				profilerSession_ = profilerSession;
			}
		}
	}
}