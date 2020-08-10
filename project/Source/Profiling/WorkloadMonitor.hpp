#pragma once

#include "Profiling/GPUMonitor.hpp"
#include "Profiling/Monitor.hpp"

namespace Profiling {
	class WorkloadMonitor : public Monitor {
		GPUMonitor gpuMonitor_;

	public:
		WorkloadMonitor(const GPUMonitor& gpuMonitor);

	protected:
		std::map<std::string, std::string> onPoll() override;
	};
}