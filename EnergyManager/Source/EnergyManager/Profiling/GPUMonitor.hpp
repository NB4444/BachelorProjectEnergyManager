#pragma once

#include "EnergyManager/Hardware/GPU.hpp"
#include "Monitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		class GPUMonitor : public Monitor {
			Hardware::GPU gpu_;

		public:
			GPUMonitor(const Hardware::GPU& gpu);

		protected:
			std::map<std::string, std::string> onPoll() override;
		};
	}
}