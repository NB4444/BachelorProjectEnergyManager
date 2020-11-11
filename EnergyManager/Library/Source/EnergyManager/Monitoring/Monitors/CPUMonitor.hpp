#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Monitoring/Monitors/CentralProcessorMonitor.hpp"

#include <map>
#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a CPU.
			 */
			class CPUMonitor : public CentralProcessorMonitor {
				/**
				 * The CPU to monitor.
				 */
				std::shared_ptr<Hardware::CPU> cpu_;

			protected:
				std::map<std::string, std::string> onPollCentralProcessor() final;

			public:
				/**
				 * Creates a new CPUMonitor.
				 * @param cpu The CPU to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				explicit CPUMonitor(const std::shared_ptr<Hardware::CPU>& cpu, const std::chrono::system_clock::duration& interval);
			};
		}
	}
}