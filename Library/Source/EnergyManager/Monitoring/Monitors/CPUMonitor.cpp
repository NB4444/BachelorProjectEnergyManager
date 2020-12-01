#include "./CPUMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> CPUMonitor::onPollCentralProcessor() {
				std::map<std::string, std::string> results;
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["turboEnabled"] = std::to_string(cpu_->getTurboEnabled()));

				return results;
			}

			CPUMonitor::CPUMonitor(const std::shared_ptr<Hardware::CPU>& cpu, const std::chrono::system_clock::duration& interval) : CentralProcessorMonitor("CPUMonitor", cpu, interval), cpu_(cpu) {
			}
		}
	}
}