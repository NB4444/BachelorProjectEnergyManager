#include "./CPUCoreMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> CPUCoreMonitor::onPollCentralProcessor() {
				std::map<std::string, std::string> results;
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["coreID"] = std::to_string(core_->getCoreID()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["cpuID"] = std::to_string(core_->getCPU()->getID()));

				return results;
			}

			CPUCoreMonitor::CPUCoreMonitor(const std::shared_ptr<Hardware::CPU::Core>& core, const std::chrono::system_clock::duration& interval)
				: CentralProcessorMonitor("CPUCoreMonitor", core, interval)
				, core_(core) {
			}
		}
	}
}