#pragma once

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Profiling/ProcessorMonitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		class GPUMonitor : public ProcessorMonitor {
			const Hardware::GPU& gpu_;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			GPUMonitor(const Hardware::GPU& gpu);
		};
	}
}