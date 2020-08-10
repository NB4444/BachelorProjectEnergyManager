#pragma once

#include "Hardware/GPU.hpp"
#include "Profiling/Monitor.hpp"

namespace Profiling {
	class GPUMonitor : public Monitor {
		Hardware::GPU gpu_;

	public:
		GPUMonitor(const Hardware::GPU& gpu);

	protected:
		std::map<std::string, std::string> onPoll() override;
	};
}