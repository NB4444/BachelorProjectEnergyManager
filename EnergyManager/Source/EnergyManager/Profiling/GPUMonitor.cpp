#include "./GPUMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

#define ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD(KEY, VALUE) ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(gpuResults[KEY] = VALUE);

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> GPUMonitor::onPoll() {
			std::map<std::string, std::string> gpuResults = {};

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

			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("applicationCoreClockRate", std::to_string(gpu_->getApplicationCoreClockRate().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("applicationMemoryClockRate", std::to_string(gpu_->getApplicationMemoryClockRate().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("autoBoostedClocksEnabled", std::to_string(gpu_->getAutoBoostedClocksEnabled()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("brand", gpu_->getBrand());
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("computeCapabilityMajorVersion", std::to_string(gpu_->getComputeCapabilityMajorVersion()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("computeCapabilityMinorVersion", std::to_string(gpu_->getComputeCapabilityMinorVersion()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("defaultPowerLimit", std::to_string(gpu_->getDefaultPowerLimit().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("defaultAutoBoostedClocksEnabled", std::to_string(gpu_->getDefaultAutoBoostedClocksEnabled()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("enforcedPowerLimit", std::to_string(gpu_->getEnforcedPowerLimit().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("fanSpeed", std::to_string(gpu_->getFanSpeed().toCombined()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("id", std::to_string(gpu_->getID()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelBlockX", std::to_string(gpu_->getKernelBlockX()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelBlockY", std::to_string(gpu_->getKernelBlockY()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelBlockZ", std::to_string(gpu_->getKernelBlockZ()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelContextID", std::to_string(gpu_->getKernelContextID()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelCorrelationID", std::to_string(gpu_->getKernelCorrelationID()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelDynamicSharedMemorySize", std::to_string(gpu_->getKernelDynamicSharedMemorySize().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelEndTimestamp", std::to_string(gpu_->getKernelEndTimestamp()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelGridX", std::to_string(gpu_->getKernelGridX()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelGridY", std::to_string(gpu_->getKernelGridY()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelGridZ", std::to_string(gpu_->getKernelGridZ()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelName", gpu_->getKernelName());
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelStartTimestamp", std::to_string(gpu_->getKernelStartTimestamp()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelStaticSharedMemorySize", std::to_string(gpu_->getKernelStaticSharedMemorySize().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("kernelStreamID", std::to_string(gpu_->getKernelStreamID()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("maximumMemoryClockRate", std::to_string(gpu_->getMaximumMemoryClockRate().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("memoryBandwidth", std::to_string(gpu_->getMemoryBandwidth().toCombined()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("memoryClockRate", std::to_string(gpu_->getMemoryClockRate().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("memoryFreeSize", std::to_string(gpu_->getMemoryFreeSize().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("memorySize", std::to_string(gpu_->getMemorySize().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("memoryUsedSize", std::to_string(gpu_->getMemoryUsedSize().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("memoryUtilizationRate", std::to_string(gpu_->getMemoryUtilizationRate().toCombined()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("multiprocessorCount", std::to_string(gpu_->getMultiprocessorCount()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("supportedMemoryClockRates", Utility::Text::join(supportedMemoryClockRates, ","));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("supportedCoreClockRates", Utility::Text::join(allSupportedCoreClockRates, ","));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("name", gpu_->getName());
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("pciELinkWidth", std::to_string(gpu_->getPCIELinkWidth().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("powerLimit", std::to_string(gpu_->getPowerLimit().toValue()));
			ENERGY_MANAGER_PROFILING_GPU_MONITOR_ADD("streamingMultiprocessorClockRate", std::to_string(gpu_->getStreamingMultiprocessorClockRate().toValue()));

			// Add the per-fan values
			for(unsigned int fan = 0;; ++fan) {
				try {
					gpuResults["fanSpeedFan" + std::to_string(fan)] = std::to_string(gpu_->getFanSpeed(fan).toCombined());
				} catch(const Utility::Exceptions::Exception& exception) {
					// Stop when we run out of fans
					break;
				}
			}

			// Get upstream values
			auto processorResults = ProcessorMonitor::onPoll();
			gpuResults.insert(processorResults.begin(), processorResults.end());

			return gpuResults;
		}

		GPUMonitor::GPUMonitor(const std::shared_ptr<Hardware::GPU>& gpu) : ProcessorMonitor("GPUMonitor", gpu), gpu_(gpu) {
		}
	}
}