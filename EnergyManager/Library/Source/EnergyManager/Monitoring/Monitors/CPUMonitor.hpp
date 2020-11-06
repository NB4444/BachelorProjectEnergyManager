#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "ProcessorMonitor.hpp"

#include <map>
#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a CPU.
			 */
			class CPUMonitor : public ProcessorMonitor {
				/**
				 * The CPU to monitor.
				 */
				std::shared_ptr<Hardware::CPU> cpu_;

				/**
				 * The user timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startUserTimespan_;

				/**
				 * The user timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreUserTimespans_;

				/**
				 * The nice timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startNiceTimespan_;

				/**
				 * The nice timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreNiceTimespans_;

				/**
				 * The system timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startSystemTimespan_;

				/**
				 * The system timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreSystemTimespans_;

				/**
				 * The idle timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startIdleTimespan_;

				/**
				 * The idle timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreIdleTimespans_;

				/**
				 * The IO wait timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startIOWaitTimespan_;

				/**
				 * The IO wait timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreIOWaitTimespans_;

				/**
				 * The interrupts timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startInterruptsTimespan_;

				/**
				 * The interrupts timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreInterruptsTimespans_;

				/**
				 * The soft interrupts timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startSoftInterruptsTimespan_;

				/**
				 * The soft interrupts timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreSoftInterruptsTimespans_;

				/**
				 * The steal timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startStealTimespan_;

				/**
				 * The steal timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreStealTimespans_;

				/**
				 * The guest timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startGuestTimespan_;

				/**
				 * The guest timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreGuestTimespans_;

				/**
				 * The guest nice timespan at the start of monitoring.
				 */
				std::chrono::system_clock::duration startGuestNiceTimespan_;

				/**
				 * The guest nice timespan per core at the start of monitoring.
				 */
				std::map<unsigned int, std::chrono::system_clock::duration> startCoreGuestNiceTimespans_;

				/**
				 * Whether the initial timespans have been measured yet.
				 */
				bool startTimespansMeasured_;

			protected:
				std::map<std::string, std::string> onPollProcessor() final;

				void onResetDevice() final;

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