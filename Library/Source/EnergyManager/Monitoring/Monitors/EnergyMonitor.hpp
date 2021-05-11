#pragma once

#include "EnergyManager/Hardware/Core.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"

enum Policies {Minmax, System, RankedMinmax };

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors the power profiler and applies energy saving strategies.
			 */
			class EnergyMonitor : public Monitor {
				/**
				 * The active state.
				 */
				enum class State {
					/**
					 * In this state the CPU and GPU are not active enough to be managed by the monitor.
					 */
					IDLE,

					/**
					 * In this state the CPU is waiting for the GPU.
					 */
					CPU_IDLE,

					/**
					 * In this state the GPU is waiting for the CPU.
					 */
					GPU_IDLE,

					/**
					 * In this state both the CPU and GPU are busy.
					 */
					BUSY,

					/**
					 * In this state both the CPU and GPU are busy, but the CPU is waiting for the GPU in a busy loop.
					 */
					CPU_BUSY_WAIT,

					/**
					 * The device is in an unknown state.
					 */
					UNKNOWN
				};

				/**
				 * The Core to monitor.
				 */
				std::shared_ptr<Hardware::Core> core_;

				/**
				 * The GPU to monitor.
				 */
				std::shared_ptr<Hardware::GPU> gpu_;

				/**
				 * Whether to run in active mode.
				 */
				bool activeMode_;

				/**
				 * The last state.
				 */
				State lastState_;

				/**
				 * Whether to use the smart policy.
				 */
				enum Policies policy_;

				/**
				 * The amount of time between halfing the frequency.
				 */
				std::chrono::system_clock::duration halfingPeriod_;

				/**
				 * The amount of time between doubling the frequency.
				 */
				std::chrono::system_clock::duration doublingPeriod_;

				/**
				 * Determines the state that is currently active.
				 * @return
				 */
				State determineState();

			protected:
				std::map<std::string, std::string> onPoll() final;

			public:
				/**
				 * Creates a new EnergyMonitor.
				 * @param core The Core to monitor.
				 * @param gpu The GPU to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 * @param halfingPeriod The amount of time between frequency halfings.
				 * @param doublingPeriod The amount of time between frequency doublings.
				 * @param activeMode Whether to run the monitor in active mode, where it changes frequency values to save energy.
				 * @param systemPolicy Whether to enable the system policy or the minmax policy.
				 */
				explicit EnergyMonitor(
					std::shared_ptr<Hardware::Core> core,
					std::shared_ptr<Hardware::GPU> gpu,
					const std::chrono::system_clock::duration& interval,
					const bool& activeMode = false,
					const std::chrono::system_clock::duration& halfingPeriod = std::chrono::system_clock::duration(0),
					const std::chrono::system_clock::duration& doublingPeriod = std::chrono::system_clock::duration(0),
					const enum Policies& policy = Minmax);
			};
		}
	}
}