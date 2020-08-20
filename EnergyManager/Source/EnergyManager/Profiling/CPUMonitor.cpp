#include "./CPUMonitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		CPUMonitor::CPUMonitor(const Hardware::CPU& cpu) : ProcessorMonitor("CPUMonitor", cpu), cpu_(cpu) {
		}
	}
}