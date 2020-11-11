#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Monitoring/Monitors/ProcessorMonitor.hpp"

#include <map>
#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a Central Processor.
			 */
			class CentralProcessorMonitor : public ProcessorMonitor {
				/**
				 * The Central Processor to monitor.
				 */
				std::shared_ptr<Hardware::CentralProcessor> processor_;

				/**
				 * The user timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startUserTimespan_;

				/**
				 * The nice timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startNiceTimespan_;

				/**
				 * The system timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startSystemTimespan_;

				/**
				 * The idle timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startIdleTimespan_;

				/**
				 * The IO wait timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startIOWaitTimespan_;

				/**
				 * The interrupts timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startInterruptsTimespan_;

				/**
				 * The soft interrupts timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startSoftInterruptsTimespan_;

				/**
				 * The steal timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startStealTimespan_;

				/**
				 * The guest timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startGuestTimespan_;

				/**
				 * The guest nice timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startGuestNiceTimespan_;

				/**
				 * Whether the initial timespans have been measured yet.
				 */
				bool startTimespansMeasured_;

			protected:
				std::map<std::string, std::string> onPollProcessor() final;

				virtual std::map<std::string, std::string> onPollCentralProcessor();

				void onResetDevice() final;

			public:
				/**
				 * Creates a new CentralProcessorMonitor.
				 * @param name The name of the Monitor.
				 * @param processor The Central Processor to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				explicit CentralProcessorMonitor(const std::string& name, const std::shared_ptr<Hardware::CentralProcessor>& processor, const std::chrono::system_clock::duration& interval);
			};
		}
	}
}