#pragma once

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Monitoring/Monitors/ProcessorMonitor.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a GPU.
			 */
			class GPUMonitor : public ProcessorMonitor {
				/**
				 * The GPU to monitor.
				 */
				std::shared_ptr<Hardware::GPU> gpu_;

				/**
				 * The timestamp of the last event that was processed.
				 */
				std::chrono::system_clock::time_point lastEventTimestamp_ = std::chrono::system_clock::now();

				/**
				 * The timestamp of the last metric that was processed.
				 */
				std::chrono::system_clock::time_point lastInstructionsPerCycleTimestamp = std::chrono::system_clock::now();

			protected:
				std::map<std::string, std::string> onPollProcessor() final;

				void onResetDevice() final;

			public:
				/**
				 * Creates a new GPUMonitor.
				 * @param gpu The GPU to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				explicit GPUMonitor(const std::shared_ptr<Hardware::GPU>& gpu, const std::chrono::system_clock::duration& interval);
			};
		}
	}
}
