#include "./ProfilerSession.hpp"

#include "EnergyManager/Monitoring/Persistence/MonitorData.hpp"

#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Persistence {
			void ProfilerSession::onSave() {
				const auto profilerSessionID = insert("ProfilerSession", { { "label", '"' + getLabel() + '"' } });
				setID(profilerSessionID);

				std::vector<std::map<std::string, std::string>> profileRows;
				for(const auto& argument : getProfile()) {
					profileRows.push_back({ { "profilerSessionID", Utility::Text::toString(profilerSessionID) },
											{ "argument", '"' + filterSQL(argument.first) + '"' },
											{ "value", '"' + filterSQL(argument.second) + '"' } });
				}
				insert("ProfilerSessionProfile", profileRows);

				for(const auto& monitorDatum : getMonitorData()) {
					monitorDatum->save();
				}
			}

			ProfilerSession::ProfilerSession(std::string label, std::map<std::string, std::string> profile, std::vector<std::shared_ptr<MonitorData>> monitorData)
				: label_(std::move(label))
				, profile_(std::move(profile))
				, monitorData_(std::move(monitorData)) {
				try {
					createTable("ProfilerSession", { { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "label", "TEXT NOT NULL" } });
				} catch(const std::runtime_error& error) {
				}
				try {
					createTable(
						"ProfilerSessionProfile",
						{ { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "profilerSessionID", "INTEGER NOT NULL" }, { "argument", "TEXT NOT NULL" }, { "value", "TEXT NOT NULL" } });
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

			std::vector<std::shared_ptr<MonitorData>> ProfilerSession::getMonitorData() const {
				return monitorData_;
			}

			void ProfilerSession::setMonitorData(const std::vector<std::shared_ptr<MonitorData>>& monitorData) {
				monitorData_ = monitorData;
			}
		}
	}
}