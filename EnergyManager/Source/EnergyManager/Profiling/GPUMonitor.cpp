#include "./GPUMonitor.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> GPUMonitor::onPoll() {
			auto gpuResults = std::map<std::string, std::string> {
				{ "computeCapabilityMajorVersion", std::to_string(gpu_->getComputeCapabilityMajorVersion()) },
				{ "computeCapabilityMinorVersion", std::to_string(gpu_->getComputeCapabilityMinorVersion()) },
				{ "coreUtilizationRate", std::to_string(gpu_->getCoreUtilizationRate()) },
				{ "fanSpeed", std::to_string(gpu_->getFanSpeed()) },
				{ "globalMemoryBandwidth", std::to_string(gpu_->getGlobalMemoryBandwidth()) },
				{ "globalMemorySize", std::to_string(gpu_->getGlobalMemorySize()) },
				{ "id", std::to_string(gpu_->getID()) },
				{ "kernelBlockX", std::to_string(gpu_->getKernelBlockX()) },
				{ "kernelBlockY", std::to_string(gpu_->getKernelBlockY()) },
				{ "kernelBlockZ", std::to_string(gpu_->getKernelBlockZ()) },
				{ "kernelContextID", std::to_string(gpu_->getKernelContextID()) },
				{ "kernelCorrelationID", std::to_string(gpu_->getKernelCorrelationID()) },
				{ "kernelDynamicSharedMemory", std::to_string(gpu_->getKernelDynamicSharedMemory()) },
				{ "kernelEndTimestamp", std::to_string(gpu_->getKernelEndTimestamp()) },
				{ "kernelGridX", std::to_string(gpu_->getKernelGridX()) },
				{ "kernelGridY", std::to_string(gpu_->getKernelGridY()) },
				{ "kernelGridZ", std::to_string(gpu_->getKernelGridZ()) },
				{ "kernelName", gpu_->getKernelName() },
				{ "kernelStartTimestamp", std::to_string(gpu_->getKernelStartTimestamp()) },
				{ "kernelStaticSharedMemory", std::to_string(gpu_->getKernelStaticSharedMemory()) },
				{ "kernelStreamID", std::to_string(gpu_->getKernelStreamID()) },
				{ "maximumMemoryClockRate", std::to_string(gpu_->getMaximumMemoryClockRate()) },
				{ "memoryClockRate", std::to_string(gpu_->getMemoryClockRate()) },
				{ "memoryUtilizationRate", std::to_string(gpu_->getMemoryUtilizationRate()) },
				{ "multiprocessorCount", std::to_string(gpu_->getMultiprocessorCount()) },
				{ "name", gpu_->getName() },
				{ "powerLimit", std::to_string(gpu_->getPowerLimit()) },
				{ "streamingMultiprocessorClockRate", std::to_string(gpu_->getStreamingMultiprocessorClockRate()) },
				{ "temperature", std::to_string(gpu_->getTemperature()) }
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