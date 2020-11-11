#include "./ApplicationMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#define ENERGY_MANAGER_MONITORING_APPLICATION_MONITOR_ADD(KEY, VALUE) ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(applicationResults[KEY] = VALUE);

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> ApplicationMonitor::onPoll() {
				std::map<std::string, std::string> applicationResults;
				ENERGY_MANAGER_MONITORING_APPLICATION_MONITOR_ADD("executableOutput", application_.getExecutableOutput());
				try {
					auto cpuAffinity = application_.getAffinity();
					std::vector<unsigned int> cpuIDs;
					std::transform(cpuAffinity.begin(), cpuAffinity.end(), std::back_inserter(cpuIDs), [](const auto& cpu) {
						return cpu->getID();
					});
					ENERGY_MANAGER_MONITORING_APPLICATION_MONITOR_ADD("cpuAffinity", Utility::Text::join(cpuIDs, ","));
				} catch(const Utility::Exceptions::Exception& exception) {
				}

				return applicationResults;
			}

			ApplicationMonitor::ApplicationMonitor(const Testing::Application& application, const std::chrono::system_clock::duration& interval)
				: Monitor("ApplicationMonitor " + application.getPath(), interval)
				, application_(application) {
			}
		}
	}
}