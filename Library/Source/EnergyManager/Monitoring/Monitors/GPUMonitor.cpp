#include "./GPUMonitor.hpp"

#include "EnergyManager/Utility/Environment.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> GPUMonitor::onPollProcessor() {
				std::map<std::string, std::string> results = {};

				std::vector<Utility::Units::Hertz> supportedMemoryClockRates = {};
				std::map<Utility::Units::Hertz, std::vector<Utility::Units::Hertz>> supportedCoreClockRates = {};
				std::set<Utility::Units::Hertz> allSupportedCoreClockRates = {};
				try {
					supportedMemoryClockRates = gpu_->getSupportedMemoryClockRates();
					for(const auto& memoryClockRate : supportedMemoryClockRates) {
						auto coreClockRate = gpu_->getSupportedCoreClockRates(memoryClockRate);

						supportedCoreClockRates[memoryClockRate] = coreClockRate;
						allSupportedCoreClockRates.insert(coreClockRate.begin(), coreClockRate.end());
					}
				} catch(const Utility::Exceptions::Exception& exception) {
				}

				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["applicationCoreClockRate"] = Utility::Text::toString(gpu_->getApplicationCoreClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["applicationMemoryClockRate"] = Utility::Text::toString(gpu_->getApplicationMemoryClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["autoBoostedClocksEnabled"] = Utility::Text::toString(gpu_->getAutoBoostedClocksEnabled()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["brand"] = gpu_->getBrand());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["computeCapabilityMajorVersion"] = Utility::Text::toString(gpu_->getComputeCapabilityMajorVersion()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["computeCapabilityMinorVersion"] = Utility::Text::toString(gpu_->getComputeCapabilityMinorVersion()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["defaultPowerLimit"] = Utility::Text::toString(gpu_->getDefaultPowerLimit().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["defaultAutoBoostedClocksEnabled"] = Utility::Text::toString(gpu_->getDefaultAutoBoostedClocksEnabled()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["enforcedPowerLimit"] = Utility::Text::toString(gpu_->getEnforcedPowerLimit().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["fanSpeed"] = Utility::Text::toString(gpu_->getFanSpeed().toCombined()));
				if(!gpu_->getKernels().empty()) {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelBlockX"] = Utility::Text::toString(gpu_->getKernels()[0].getBlockX()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelBlockY"] = Utility::Text::toString(gpu_->getKernels()[0].getBlockY()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelBlockZ"] = Utility::Text::toString(gpu_->getKernels()[0].getBlockZ()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelContextID"] = Utility::Text::toString(gpu_->getKernels()[0].getContextID()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelCorrelationID"] = Utility::Text::toString(gpu_->getKernels()[0].getCorrelationID()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["kernelDynamicSharedMemorySize"] = Utility::Text::toString(gpu_->getKernels()[0].getDynamicSharedMemorySize().toValue()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelEndTimestamp"] = Utility::Text::toString(gpu_->getKernels()[0].getEndTimestamp()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelGridX"] = Utility::Text::toString(gpu_->getKernels()[0].getGridX()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelGridY"] = Utility::Text::toString(gpu_->getKernels()[0].getGridY()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelGridZ"] = Utility::Text::toString(gpu_->getKernels()[0].getGridZ()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelName"] = gpu_->getKernels()[0].getName());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelStartTimestamp"] = Utility::Text::toString(gpu_->getKernels()[0].getStartTimestamp()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelStaticSharedMemorySize"] = Utility::Text::toString(gpu_->getKernels()[0].getStaticSharedMemorySize().toValue()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelStreamID"] = Utility::Text::toString(gpu_->getKernels()[0].getStreamID()));
				}
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["maximumMemoryClockRate"] = Utility::Text::toString(gpu_->getMaximumMemoryClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryBandwidth"] = Utility::Text::toString(gpu_->getMemoryBandwidth().toCombined()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryClockRate"] = Utility::Text::toString(gpu_->getMemoryClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryFreeSize"] = Utility::Text::toString(gpu_->getMemoryFreeSize().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memorySize"] = Utility::Text::toString(gpu_->getMemorySize().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryUsedSize"] = Utility::Text::toString(gpu_->getMemoryUsedSize().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryUtilizationRate"] = Utility::Text::toString(gpu_->getMemoryUtilizationRate().toCombined()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["multiprocessorCount"] = Utility::Text::toString(gpu_->getMultiprocessorCount()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["supportedMemoryClockRates"] = Utility::Text::join(supportedMemoryClockRates, ","));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["supportedCoreClockRates"] = Utility::Text::join(allSupportedCoreClockRates, ","));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["name"] = gpu_->getName());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["pciELinkWidth"] = Utility::Text::toString(gpu_->getPCIELinkWidth().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["powerLimit"] = Utility::Text::toString(gpu_->getPowerLimit().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["streamingMultiprocessorClockRate"] = Utility::Text::toString(gpu_->getStreamingMultiprocessorClockRate().toValue()));

				// Add the per-fan values
				for(unsigned int fan = 0;; ++fan) {
					try {
						results["fanSpeedFan" + Utility::Text::toString(fan)] = Utility::Text::toString(gpu_->getFanSpeed(fan).toCombined());
					} catch(const Utility::Exceptions::Exception& exception) {
						// Stop when we run out of fans
						break;
					}
				}

				// Add the new events
				logTrace("Looking for GPU events...");
				for(const auto& timestampEvents : gpu_->getEvents()) {
					const auto& timestamp = timestampEvents.first;
					const auto& events = timestampEvents.second;

					// Only process new events
					if(timestamp > lastEventTimestamp_) {
						for(const auto& event : events) {
							const auto& eventName = event.first;
							const auto& eventSite = event.second;

							logTrace("Processing event that occurred at %s with name %s", Utility::Text::formatTimestamp(timestamp).c_str(), eventName.c_str());

							// Check if there are already events at that timestamp
							std::vector<std::string> currentEvents = {};
							if(hasVariable(timestamp, "events")) {
								// If so, retrieve those
								currentEvents = Utility::Text::splitToVector(getVariable(timestamp, "events"), ",");
							}

							// Append the new event
							currentEvents.push_back(eventName + "(" + (eventSite == Hardware::GPU::EventSite::ENTER ? "ENTER" : "EXIT") + ")");

							// Store the variable
							setVariable(timestamp, "events", Utility::Text::join(currentEvents, ","));

							// Update the last timestamp
							lastEventTimestamp_ = timestamp;
						}
					} else {
						logTrace("Outdated events, ignoring...");
					}
				}

				// Add the IPC data
				logTrace("Looking for GPU IPC...");
				for(const auto& timestampInstructionsPerCycle : gpu_->getInstructionsPerCycle()) {
					const auto& timestamp = timestampInstructionsPerCycle.first;
					const auto& instructionsPerCycle = timestampInstructionsPerCycle.second;

					// Only process new events
					if(timestamp > lastInstructionsPerCycleTimestamp) {
						logTrace("Processing IPC that occurred at %s with value %f", Utility::Text::formatTimestamp(timestamp).c_str(), instructionsPerCycle.toCombined());

						// Store the variable
						setVariable(timestamp, "instructionsPerCycle", Utility::Text::toString(instructionsPerCycle.toCombined()));

						// Update the last timestamp
						lastInstructionsPerCycleTimestamp = timestamp;
					} else {
						logTrace("Outdated IPC, ignoring...");
					}
				}

				return results;
			}

			GPUMonitor::GPUMonitor(const std::shared_ptr<Hardware::GPU>& gpu, const std::chrono::system_clock::duration& interval) : ProcessorMonitor("GPUMonitor", gpu, interval), gpu_(gpu) {
			}
		}
	}
}