#include "./GPUMonitor.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> GPUMonitor::onPoll() {
			auto gpuResults = std::map<std::string, std::string> {
				{ "computeCapabilityMajorVersion", std::to_string(gpu_->getComputeCapabilityMajorVersion()) },
				{ "computeCapabilityMinorVersion", std::to_string(gpu_->getComputeCapabilityMinorVersion()) },
				{ "coreUtilizationRate", std::to_string(gpu_->getCoreUtilizationRate().toCombined()) },
				{ "fanSpeed", std::to_string(gpu_->getFanSpeed().toCombined()) },
				{ "globalMemoryBandwidth", std::to_string(gpu_->getGlobalMemoryBandwidth().toCombined()) },
				{ "globalMemorySize", std::to_string(gpu_->getGlobalMemorySize().toValue()) },
				{ "id", std::to_string(gpu_->getID()) },
				{ "kernelBlockX", std::to_string(gpu_->getKernelBlockX()) },
				{ "kernelBlockY", std::to_string(gpu_->getKernelBlockY()) },
				{ "kernelBlockZ", std::to_string(gpu_->getKernelBlockZ()) },
				{ "kernelContextID", std::to_string(gpu_->getKernelContextID()) },
				{ "kernelCorrelationID", std::to_string(gpu_->getKernelCorrelationID()) },
				{ "kernelDynamicSharedMemory", std::to_string(gpu_->getKernelDynamicSharedMemory().toValue()) },
				{ "kernelEndTimestamp", std::to_string(gpu_->getKernelEndTimestamp()) },
				{ "kernelGridX", std::to_string(gpu_->getKernelGridX()) },
				{ "kernelGridY", std::to_string(gpu_->getKernelGridY()) },
				{ "kernelGridZ", std::to_string(gpu_->getKernelGridZ()) },
				{ "kernelName", gpu_->getKernelName() },
				{ "kernelStartTimestamp", std::to_string(gpu_->getKernelStartTimestamp()) },
				{ "kernelStaticSharedMemory", std::to_string(gpu_->getKernelStaticSharedMemory().toValue()) },
				{ "kernelStreamID", std::to_string(gpu_->getKernelStreamID()) },
				{ "maximumMemoryClockRate", std::to_string(gpu_->getMaximumMemoryClockRate().toValue()) },
				{ "memoryClockRate", std::to_string(gpu_->getMemoryClockRate().toValue()) },
				{ "memoryUtilizationRate", std::to_string(gpu_->getMemoryUtilizationRate().toCombined()) },
				{ "multiprocessorCount", std::to_string(gpu_->getMultiprocessorCount()) },
				{ "name", gpu_->getName() },
				{ "powerLimit", std::to_string(gpu_->getPowerLimit().toValue()) },
				{ "streamingMultiprocessorClockRate", std::to_string(gpu_->getStreamingMultiprocessorClockRate().toValue()) },
				{ "temperature", std::to_string(gpu_->getTemperature().toValue()) }
			};

			// Get upstream values
			auto processorResults = ProcessorMonitor::onPoll();
			gpuResults.insert(processorResults.begin(), processorResults.end());

			return gpuResults;
		}

		GPUMonitor::GPUMonitor(const std::shared_ptr<Hardware::GPU>& gpu) : ProcessorMonitor("GPUMonitor", gpu), gpu_(gpu) {
		}
	}
}