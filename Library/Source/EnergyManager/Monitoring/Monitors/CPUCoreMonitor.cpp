#include "./CPUCoreMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> CPUCoreMonitor::onPollCentralProcessor() {
				std::map<std::string, std::string> results;
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["coreID"] = Utility::Text::toString(core_->getCoreID()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["cpuID"] = Utility::Text::toString(core_->getCPU()->getID()));

				return results;
			}

			CPUCoreMonitor::CPUCoreMonitor(const std::shared_ptr<Hardware::Core>& core, const std::chrono::system_clock::duration& interval)
				: CentralProcessorMonitor("CPUCoreMonitor", core, interval)
				, core_(core) {
			}
		}
	}
}