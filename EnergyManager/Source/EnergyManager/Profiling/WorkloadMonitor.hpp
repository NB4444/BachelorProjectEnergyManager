#pragma once

#include "GPUMonitor.hpp"
#include "Monitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		class WorkloadMonitor : public Monitor {
			GPUMonitor gpuMonitor_;

		public:
			WorkloadMonitor(const GPUMonitor& gpuMonitor);

		protected:
			std::map<std::string, std::string> onPoll() override;
		};
	}
}