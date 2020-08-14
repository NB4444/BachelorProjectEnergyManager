#include "./CPUMonitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		CPUMonitor::CPUMonitor(const Hardware::CPU& cpu) : Monitor("CPUMonitor"), cpu_(cpu) {
		}

		std::map<std::string, std::string> CPUMonitor::onPoll() {
			return {
				{ "coreClockRate", std::to_string(cpu_.getCoreClockRate()) },
				{ "maximumCoreClockRate", std::to_string(cpu_.getMaximumCoreClockRate()) },
			};
		}
	}
}