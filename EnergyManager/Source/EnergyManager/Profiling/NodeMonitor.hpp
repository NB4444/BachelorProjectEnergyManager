#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Profiling/Monitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		class NodeMonitor : public Monitor {
			const Hardware::CPU& cpu_;

			const Hardware::GPU& gpu_;

			float startEnergyConsumption_ = 0;

			bool startEnergyConsumptionMeasured_ = false;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			NodeMonitor(const Hardware::CPU& cpu, const Hardware::GPU& gpu);
		};
	}
}