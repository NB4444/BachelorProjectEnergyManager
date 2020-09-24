#pragma once

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Monitoring/ProcessorMonitor.hpp"

namespace EnergyManager {
	namespace Monitoring {
		class GPUMonitor : public ProcessorMonitor {
			std::shared_ptr<Hardware::GPU> gpu_;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			static void initialize();

			GPUMonitor(const std::shared_ptr<Hardware::GPU>& gpu);
		};
	}
}