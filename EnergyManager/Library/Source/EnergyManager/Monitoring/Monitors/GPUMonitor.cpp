#include "./GPUMonitor.hpp"

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

				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["applicationCoreClockRate"] = std::to_string(gpu_->getApplicationCoreClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["applicationMemoryClockRate"] = std::to_string(gpu_->getApplicationMemoryClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["autoBoostedClocksEnabled"] = std::to_string(gpu_->getAutoBoostedClocksEnabled()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["brand"] = gpu_->getBrand());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["computeCapabilityMajorVersion"] = std::to_string(gpu_->getComputeCapabilityMajorVersion()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["computeCapabilityMinorVersion"] = std::to_string(gpu_->getComputeCapabilityMinorVersion()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["defaultPowerLimit"] = std::to_string(gpu_->getDefaultPowerLimit().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["defaultAutoBoostedClocksEnabled"] = std::to_string(gpu_->getDefaultAutoBoostedClocksEnabled()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["enforcedPowerLimit"] = std::to_string(gpu_->getEnforcedPowerLimit().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["fanSpeed"] = std::to_string(gpu_->getFanSpeed().toCombined()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelBlockX"] = std::to_string(gpu_->getKernelBlockX()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelBlockY"] = std::to_string(gpu_->getKernelBlockY()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelBlockZ"] = std::to_string(gpu_->getKernelBlockZ()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelContextID"] = std::to_string(gpu_->getKernelContextID()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelCorrelationID"] = std::to_string(gpu_->getKernelCorrelationID()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelDynamicSharedMemorySize"] = std::to_string(gpu_->getKernelDynamicSharedMemorySize().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelEndTimestamp"] = std::to_string(gpu_->getKernelEndTimestamp()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelGridX"] = std::to_string(gpu_->getKernelGridX()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelGridY"] = std::to_string(gpu_->getKernelGridY()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelGridZ"] = std::to_string(gpu_->getKernelGridZ()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelName"] = gpu_->getKernelName());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelStartTimestamp"] = std::to_string(gpu_->getKernelStartTimestamp()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelStaticSharedMemorySize"] = std::to_string(gpu_->getKernelStaticSharedMemorySize().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["kernelStreamID"] = std::to_string(gpu_->getKernelStreamID()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["maximumMemoryClockRate"] = std::to_string(gpu_->getMaximumMemoryClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryBandwidth"] = std::to_string(gpu_->getMemoryBandwidth().toCombined()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryClockRate"] = std::to_string(gpu_->getMemoryClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryFreeSize"] = std::to_string(gpu_->getMemoryFreeSize().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memorySize"] = std::to_string(gpu_->getMemorySize().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryUsedSize"] = std::to_string(gpu_->getMemoryUsedSize().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["memoryUtilizationRate"] = std::to_string(gpu_->getMemoryUtilizationRate().toCombined()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["multiprocessorCount"] = std::to_string(gpu_->getMultiprocessorCount()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["supportedMemoryClockRates"] = Utility::Text::join(supportedMemoryClockRates, ","));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["supportedCoreClockRates"] = Utility::Text::join(allSupportedCoreClockRates, ","));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["name"] = gpu_->getName());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["pciELinkWidth"] = std::to_string(gpu_->getPCIELinkWidth().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["powerLimit"] = std::to_string(gpu_->getPowerLimit().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["streamingMultiprocessorClockRate"] = std::to_string(gpu_->getStreamingMultiprocessorClockRate().toValue()));

				// Add the per-fan values
				for(unsigned int fan = 0;; ++fan) {
					try {
						results["fanSpeedFan" + std::to_string(fan)] = std::to_string(gpu_->getFanSpeed(fan).toCombined());
					} catch(const Utility::Exceptions::Exception& exception) {
						// Stop when we run out of fans
						break;
					}
				}

				return results;
			}

			GPUMonitor::GPUMonitor(const std::shared_ptr<Hardware::GPU>& gpu, const std::chrono::system_clock::duration& interval) : ProcessorMonitor("GPUMonitor", gpu, interval), gpu_(gpu) {
			}
		}
	}
}