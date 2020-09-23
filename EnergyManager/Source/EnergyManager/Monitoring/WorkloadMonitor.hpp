#pragma once

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Monitoring/ProcessorMonitor.hpp"

namespace EnergyManager {
	namespace Monitoring {
		class WorkloadMonitor : public Monitor {
			std::shared_ptr<Hardware::GPU> gpu_;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			WorkloadMonitor(const std::shared_ptr<Hardware::GPU>& gpu);
		};
	}
}