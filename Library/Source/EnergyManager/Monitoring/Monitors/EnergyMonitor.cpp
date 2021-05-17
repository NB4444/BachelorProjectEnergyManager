#include "./EnergyMonitor.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/Core.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <utility>

#define SCALING_CLOCKRATE(device, scale) \
	((device->getMaximumCoreClockRate().toValue() - device->getMinimumCoreClockRate().toValue()) * scale + device->getMinimumCoreClockRate().toValue())
namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			EnergyMonitor::State EnergyMonitor::determineState() {
				// Define the boundary above which values are considered "high" (inclusive)
				const auto highBoundary = 10;

				// First determine the utilization rates
				enum class Utilization { HIGH, LOW };
				auto coreUtilizationRate = core_->getCoreUtilizationRate().getUnit() < highBoundary ? Utilization::LOW : Utilization::HIGH;
				auto gpuUtilizationRate = gpu_->getCoreUtilizationRate().getUnit() < highBoundary ? Utilization::LOW : Utilization::HIGH;

				// Determine if there is a synchronize active between the CPU and GPU
				bool cudaEventSynchronize = false;
				bool cudaDeviceSynchronize = false;
				bool cuEventSynchronize = false;
				bool cuDeviceSynchronize = false;
				for(const auto& timestampEvents : gpu_->getEvents()) {
					const auto& timestamp = timestampEvents.first;
					const auto& events = timestampEvents.second;

					// Process the events
					for(const auto& event : events) {
						const auto& eventName = Utility::Text::trim(event.first);
						const auto& eventSite = event.second;

						if(eventName == "cudaDeviceSynchronize") {
							if(eventSite == Hardware::GPU::EventSite::ENTER) {
								cudaDeviceSynchronize = true;
							} else {
								cudaDeviceSynchronize = false;
							}
						}

						if(eventName == "cuDeviceSynchronize") {
							if(eventSite == Hardware::GPU::EventSite::ENTER) {
								cuDeviceSynchronize = true;
							} else {
								cuDeviceSynchronize = false;
							}
						}

						if(eventName == "cudaEventSynchronize") {
							if(eventSite == Hardware::GPU::EventSite::ENTER) {
								cudaEventSynchronize = true;
							} else {
								cudaEventSynchronize = false;
							}
						}

						if(eventName == "cuEventSynchronize") {
							if(eventSite == Hardware::GPU::EventSite::ENTER) {
								cuEventSynchronize = true;
							} else {
								cuEventSynchronize = false;
							}
						}
					}
				}
				bool synchronizationActive = cudaDeviceSynchronize || cuDeviceSynchronize || cudaEventSynchronize || cuEventSynchronize;

				// Determine the active state
				if(coreUtilizationRate == Utilization::LOW && gpuUtilizationRate == Utilization::LOW) {
					return State::IDLE;
				} else if(coreUtilizationRate == Utilization::LOW && gpuUtilizationRate == Utilization::HIGH) {
					return State::CPU_IDLE;
				} else if(coreUtilizationRate == Utilization::HIGH && gpuUtilizationRate == Utilization::LOW) {
					return State::GPU_IDLE;
				} else if(coreUtilizationRate == Utilization::HIGH && gpuUtilizationRate == Utilization::HIGH) {
					if(synchronizationActive) {
						return State::CPU_BUSY_WAIT;
					} else {
						return State::BUSY;
					}
				}

				return State::UNKNOWN;
			}

			std::map<std::string, std::string> EnergyMonitor::onPoll() {
				/**
				 * To calculate the inactive scaling:
				 * x=scaling factor
				 * y=monitor intervals in halfing period
				 * z=clock speed
				 * Solve for x: x^y*z=0.5z
				 */
				const auto halfingPeriodMonitorIntervals = halfingPeriod_ / getInterval();
				const auto inactiveScaling = std::pow(2.0, -1.0 / halfingPeriodMonitorIntervals);

				/**
				 * To calculate the busy scaling:
				 * x=scaling factor
				 * y=monitor intervals in doubling period
				 * z=clock speed
				 * Solve for x: x^y*z=2z
				 */
				const auto doublingPeriodsMonitorIntervals = doublingPeriod_ / getInterval();
				const auto busyScaling = std::pow(2.0, 1.0 / doublingPeriodsMonitorIntervals);

				// Functions responsible for scaling frequencies
				const auto scaleCPUDown = [&] {
					const auto clockRate = std::max(core_->getMinimumCoreClockRate().toValue(), inactiveScaling * core_->getCoreClockRate().toValue());
					core_->getCPU()->setTurboEnabled(false);
					core_->setCoreClockRate(clockRate, clockRate);
					core_->getCPU()->setCoreClockRate(clockRate, clockRate);
				};
				const auto scaleCPUUp = [&] {
					const auto clockRate = std::min(core_->getMaximumCoreClockRate().toValue(), busyScaling * core_->getCoreClockRate().toValue());
					core_->getCPU()->setTurboEnabled(false);
					core_->setCoreClockRate(clockRate, clockRate);
					core_->getCPU()->setCoreClockRate(clockRate, clockRate);
				};
				const auto scaleGPUDown = [&] {
					const auto clockRate = std::max(gpu_->getMinimumCoreClockRate().toValue(), inactiveScaling * gpu_->getCoreClockRate().toValue());
					gpu_->setCoreClockRate(clockRate, clockRate);
				};
				const auto scaleGPUUp = [&] {
					const auto clockRate = std::min(gpu_->getMaximumCoreClockRate().toValue(), busyScaling * gpu_->getCoreClockRate().toValue());
					gpu_->setCoreClockRate(clockRate, clockRate);
				};
				
				const auto scaleGPURankedUp = [&] {
					Utility::Units::Hertz clockRate;
					if (gpu_->getCoreUtilizationRate() <= 25) {
						clockRate = std::min(SCALING_CLOCKRATE(gpu_, 0.25), busyScaling * gpu_->getCoreClockRate().toValue());
					} else if(gpu_->getCoreUtilizationRate() <= 50) {
						clockRate = std::min(SCALING_CLOCKRATE(gpu_, 0.5), busyScaling * gpu_->getCoreClockRate().toValue());
					} else if(gpu_->getCoreUtilizationRate() <= 75) {
						clockRate = std::min(SCALING_CLOCKRATE(gpu_, 0.75), busyScaling * gpu_->getCoreClockRate().toValue());
					} else {
						clockRate = std::min(gpu_->getMaximumCoreClockRate().toValue(), busyScaling * gpu_->getCoreClockRate().toValue());
					}
					
					gpu_->setCoreClockRate(clockRate, clockRate);
				};
				
				const auto scaleCPURankedUp = [&] {
					Utility::Units::Hertz clockRate;
					if (core_->getCoreUtilizationRate() <= 25) {
						clockRate = std::min(SCALING_CLOCKRATE(core_, 0.25), busyScaling * core_->getCoreClockRate().toValue());
					} else if(core_->getCoreUtilizationRate() <= 50) {
						clockRate = std::min(SCALING_CLOCKRATE(core_, 0.5), busyScaling * core_->getCoreClockRate().toValue());
					} else if(core_->getCoreUtilizationRate() <= 75) {
						clockRate = std::min(SCALING_CLOCKRATE(core_, 0.75), busyScaling * core_->getCoreClockRate().toValue());
					} else {
						clockRate = std::min(core_->getMaximumCoreClockRate().toValue(), busyScaling * core_->getCoreClockRate().toValue());
					}
					
					core_->getCPU()->setTurboEnabled(false);
					core_->setCoreClockRate(clockRate, clockRate);
					core_->getCPU()->setCoreClockRate(clockRate, clockRate);
				};
				
				const auto scaleGPUScaledUp = [&] {
					auto clockRate = std::min(SCALING_CLOCKRATE(gpu_, gpu_->getCoreUtilizationRate().getUnit() * 0.01), busyScaling * gpu_->getCoreClockRate().toValue());
					
					gpu_->setCoreClockRate(clockRate, clockRate);
				};
				
				const auto scaleCPUScaledUp = [&] {
					auto clockRate = std::min(SCALING_CLOCKRATE(core_, core_->getCoreUtilizationRate().getUnit() * 0.01), busyScaling * core_->getCoreClockRate().toValue());
					
					core_->getCPU()->setTurboEnabled(false);
					core_->setCoreClockRate(clockRate, clockRate);
					core_->getCPU()->setCoreClockRate(clockRate, clockRate);
				};

				std::map<std::string, std::string> results = {};

				// Determine the state
				auto currentState = determineState();

				// Store the state
				std::string stateString;
				switch(currentState) {
					case State::CPU_IDLE:
						stateString = "CPU_IDLE";
						break;
					case State::GPU_IDLE:
						stateString = "GPU_IDLE";
						break;
					case State::BUSY:
						stateString = "BUSY";
						break;
					case State::CPU_BUSY_WAIT:
						stateString = "CPU_BUSY_WAIT";
						break;
					case State::IDLE:
						stateString = "IDLE";
						break;
					case State::UNKNOWN:
					default:
						stateString = "UNKNOWN";
						break;
				}
				results["state"] = stateString;

				// Take action if the state changes
				if(activeMode_ && currentState != lastState_) {
					// Reset configured values to the defaults
					Utility::Units::Hertz clockRateCPU, clockRateGPU;
					switch(policy_) {
						case ScalingMinmax:
							clockRateCPU = SCALING_CLOCKRATE(core_, core_->getCoreUtilizationRate().getUnit() * 0.01);
							core_->getCPU()->setTurboEnabled(true);
							core_->setCoreClockRate(clockRateCPU, clockRateCPU);
							core_->getCPU()->setCoreClockRate(clockRateCPU, clockRateCPU);
							
							clockRateCPU = SCALING_CLOCKRATE(gpu_, gpu_->getCoreUtilizationRate().getUnit() * 0.01);
							gpu_->setCoreClockRate(clockRateGPU, clockRateGPU);
							break;
						case RankedMinmax:
							if(core_->getCoreUtilizationRate() <= 25) {
								clockRateCPU = SCALING_CLOCKRATE(core_, 0.25);
							} else if (core_->getCoreUtilizationRate() <= 50) {
								clockRateCPU = SCALING_CLOCKRATE(core_, 0.5);
							} else if (core_->getCoreUtilizationRate() <= 75) {
								clockRateCPU = SCALING_CLOCKRATE(core_, 0.75);
							} else {
								clockRateCPU = core_->getMaximumCoreClockRate().toValue();
							}
							core_->getCPU()->setTurboEnabled(true);
							core_->setCoreClockRate(clockRateCPU, clockRateCPU);
							core_->getCPU()->setCoreClockRate(clockRateCPU, clockRateCPU);
							
							if (gpu_->getCoreUtilizationRate() <= 25) {
								clockRateGPU = SCALING_CLOCKRATE(gpu_, 0.25);
							} else if(gpu_->getCoreUtilizationRate() <= 50) {
								clockRateGPU = SCALING_CLOCKRATE(gpu_, 0.5);
							} else if(gpu_->getCoreUtilizationRate() <= 75) {
								clockRateGPU = SCALING_CLOCKRATE(gpu_, 0.75);
							} else {
								clockRateGPU = gpu_->getMaximumCoreClockRate().toValue();
							}

							gpu_->setCoreClockRate(clockRateGPU, clockRateGPU);
							break;
						case Minmax:
							core_->getCPU()->setTurboEnabled(true);
							core_->setCoreClockRate(core_->getMaximumCoreClockRate(), core_->getMaximumCoreClockRate());
							core_->getCPU()->setCoreClockRate(core_->getMaximumCoreClockRate(), core_->getCPU()->getMaximumCoreClockRate());
							gpu_->setCoreClockRate(gpu_->getMaximumCoreClockRate(), gpu_->getMaximumCoreClockRate());
							break;
						case System:
							core_->getCPU()->setTurboEnabled(true);
							core_->resetCoreClockRate();
							core_->getCPU()->resetCoreClockRate();
							gpu_->resetCoreClockRate();
							break;
					}

					// Apply the new configuration
					switch(currentState) {
						case State::CPU_IDLE:
						case State::CPU_BUSY_WAIT:
							switch(policy_) {
								case Minmax:
								case System:
									scaleCPUDown();
									scaleGPUUp();
									break;
								case RankedMinmax:
									scaleCPUDown();
									scaleGPURankedUp();
									break;
								case ScalingMinmax:
									scaleCPUDown();
									scaleGPUScaledUp();
									break;
							}
							break;
						case State::GPU_IDLE:
							switch(policy_) {
								case Minmax:
								case System:
									scaleGPUDown();
									scaleCPUUp();
									break;
								case RankedMinmax:
									scaleGPUDown();
									scaleGPURankedUp();
									break;
								case ScalingMinmax:
									scaleGPUDown();
									scaleGPUScaledUp();
									break;
							}
							break;
						case State::BUSY:
							switch(policy_) {
								case Minmax:
									scaleCPUUp();
									scaleGPUUp();
									break;
								case RankedMinmax:
									scaleGPURankedUp();
									scaleCPURankedUp();
									break;
								case ScalingMinmax:
									scaleGPUScaledUp();
									scaleCPUScaledUp();
									break;
								case System:
									break;
							}
							break;
						case State::IDLE:
							switch(policy_) {
								case Minmax:
								case RankedMinmax:
								case ScalingMinmax:
									scaleCPUDown();
									scaleGPUDown();
									break;
								case System:
									break;
							}
							break;
						case State::UNKNOWN:
						default:
							// In unknown states no action is taken
							break;
					}
				}
				
				//if(policy_ == ScalingMinmax) {
				//	switch(currentState) {
				//		case State::BUSY:
				//			scaleGPUScaledUp();
				//			scaleCPUScaledUp();
				//			break;
				//		case State::GPU_IDLE:
				//			scaleCPUScaledUp();
				//			break;
				//		case State::CPU_IDLE:
				//		case State::CPU_BUSY_WAIT:
				//			scaleGPUScaledUp();
				//			break;
				//		case State::IDLE:
				//		case State::UNKNOWN:
				//			break;
				//	}
				//}

				// Store the current state for the next poll
				lastState_ = currentState;

				return results;
			}

			EnergyMonitor::EnergyMonitor(
				std::shared_ptr<Hardware::Core> core,
				std::shared_ptr<Hardware::GPU> gpu,
				const std::chrono::system_clock::duration& interval,
				const bool& activeMode,
				const std::chrono::system_clock::duration& halfingPeriod,
				const std::chrono::system_clock::duration& doublingPeriod,
				const enum Policies& policy)
				: Monitor("EnergyMonitor", interval)
				, halfingPeriod_(halfingPeriod)
				, doublingPeriod_(doublingPeriod)
				, core_(std::move(core))
				, gpu_(std::move(gpu))
				, activeMode_(activeMode)
				, policy_(policy) {
			}
		}
	}
}