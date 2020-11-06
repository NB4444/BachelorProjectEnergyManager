#include "./MonitorData.hpp"

#include "EnergyManager/Monitoring/Persistence/ProfilerSession.hpp"

#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Persistence {
			void MonitorData::onSave() {
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

						monitorData.push_back({ { "profilerSessionID", std::to_string(getProfilerSession()->getID()) },
												{ "monitorName", '"' + filterSQL(getMonitorName()) + '"' },
												{ "timestamp", timestamp },
												{ "name", '"' + filterSQL(name) + '"' },
												{ "value", '"' + filterSQL(value) + '"' } });
					}
				}

				insert("MonitorData", monitorData);
			}

			MonitorData::MonitorData(
				std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> monitorData,
				std::string monitorName,
				std::shared_ptr<ProfilerSession> profilerSession)
				: monitorData_(std::move(monitorData))
				, monitorName_(std::move(monitorName))
				, profilerSession_(std::move(profilerSession)) {
				try {
					createTable(
						"MonitorData",
						{ { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" },
						  { "profilerSessionID", "INTEGER NOT NULL" },
						  { "monitorName", "TEXT NOT NULL" },
						  { "timestamp", "INTEGER NOT NULL" },
						  { "name", "TEXT NOT NULL" },
						  { "value", "TEXT" } });
				} catch(const std::runtime_error& error) {
				}
			}

			std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> MonitorData::getMonitorData() const {
				return monitorData_;
			}

			void MonitorData::setMonitorData(const std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>& monitorData) {
				monitorData_ = monitorData;
			}

			std::string MonitorData::getMonitorName() const {
				return monitorName_;
			}

			void MonitorData::setMonitorName(const std::string& monitorName) {
				monitorName_ = monitorName;
			}

			std::shared_ptr<ProfilerSession> MonitorData::getProfilerSession() const {
				return profilerSession_;
			}

			void MonitorData::setProfilerSession(const std::shared_ptr<ProfilerSession>& profilerSession) {
				profilerSession_ = profilerSession;
			}
		}
	}
}