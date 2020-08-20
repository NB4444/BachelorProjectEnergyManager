#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Profiling/ProcessorMonitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		class CPUMonitor : public ProcessorMonitor {
			const Hardware::CPU& cpu_;

		public:
			CPUMonitor(const Hardware::CPU& cpu);
		};
	}
}