#pragma once

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Profiling/Monitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		class GPUMonitor : public Monitor {
			const Hardware::GPU& gpu_;

			float totalPowerConsumption_ = 0;

		public:
			GPUMonitor(const Hardware::GPU& gpu);

		protected:
			std::map<std::string, std::string> onPoll() override;
		};
	}
}