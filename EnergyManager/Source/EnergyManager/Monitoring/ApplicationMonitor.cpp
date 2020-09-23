#include "./ApplicationMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#define ENERGY_MANAGER_PROFILING_APPLICATION_MONITOR_ADD(KEY, VALUE) ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(applicationResults[KEY] = VALUE);

namespace EnergyManager {
	namespace Monitoring {
		std::map<std::string, std::string> ApplicationMonitor::onPoll() {
			auto cpuAffinity = application_.getCPUAffinity();
			std::vector<unsigned int> cpuIDs;
			std::transform(cpuAffinity.begin(), cpuAffinity.end(), std::back_inserter(cpuIDs), [] (const auto& cpu) {
				return cpu->getID();
			});

			std::map<std::string, std::string> applicationResults;
			ENERGY_MANAGER_PROFILING_APPLICATION_MONITOR_ADD("executableOutput", application_.getExecutableOutput());
			ENERGY_MANAGER_PROFILING_APPLICATION_MONITOR_ADD("cpuAffinity", Utility::Text::join(cpuIDs, ","));

			return applicationResults;
		}

		ApplicationMonitor::ApplicationMonitor(const Application& application) : Monitor("ApplicationMonitor"), application_(application) {
		}
	}
}