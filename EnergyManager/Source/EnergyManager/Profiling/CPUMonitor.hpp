#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Profiling/Monitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		class CPUMonitor : public Monitor {
			const Hardware::CPU& cpu_;

		public:
			CPUMonitor(const Hardware::CPU& cpu);

		protected:
			std::map<std::string, std::string> onPoll() override;
		};
	}
}