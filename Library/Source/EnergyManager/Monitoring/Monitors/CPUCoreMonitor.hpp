#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Monitoring/Monitors/CentralProcessorMonitor.hpp"

#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a CPU Core.
			 */
			class CPUCoreMonitor : public CentralProcessorMonitor {
				/**
				 * The CPU Core to monitor.
				 */
				std::shared_ptr<Hardware::CPU::Core> core_;

			protected:
				std::map<std::string, std::string> onPollCentralProcessor() final;

			public:
				/**
				 * Creates a new CPUCoreMonitor.
				 * @param core The CPU Core to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				explicit CPUCoreMonitor(const std::shared_ptr<Hardware::CPU::Core>& core, const std::chrono::system_clock::duration& interval);
			};
		}
	}
}