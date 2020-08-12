#include "./GPUMonitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		GPUMonitor::GPUMonitor(const Hardware::GPU& gpu)
			: Monitor("GPUMonitor")
			, gpu_(gpu) {
		}

		std::map<std::string, std::string> GPUMonitor::onPoll() {
			return {
				{ "coreClockRate", std::to_string(gpu_.getCoreClockRate()) },
				{ "fanSpeed", std::to_string(gpu_.getFanSpeed()) },
				{ "kernelBlockX", std::to_string(gpu_.getKernelBlockX()) },
				{ "kernelBlockY", std::to_string(gpu_.getKernelBlockY()) },
				{ "kernelBlockZ", std::to_string(gpu_.getKernelBlockZ()) },
				{ "kernelContextID", std::to_string(gpu_.getKernelContextID()) },
				{ "kernelCorrelationID", std::to_string(gpu_.getKernelCorrelationID()) },
				{ "kernelDynamicSharedMemory", std::to_string(gpu_.getKernelDynamicSharedMemory()) },
				{ "kernelEndTimestamp", std::to_string(gpu_.getKernelEndTimestamp()) },
				{ "kernelGridX", std::to_string(gpu_.getKernelGridX()) },
				{ "kernelGridY", std::to_string(gpu_.getKernelGridY()) },
				{ "kernelGridZ", std::to_string(gpu_.getKernelGridZ()) },
				{ "kernelName", gpu_.getKernelName() },
				{ "kernelStartTimestamp", std::to_string(gpu_.getKernelStartTimestamp()) },
				{ "kernelStaticSharedMemory", std::to_string(gpu_.getKernelStaticSharedMemory()) },
				{ "kernelStreamID", std::to_string(gpu_.getKernelStreamID()) },
				{ "memoryClock", std::to_string(gpu_.getMemoryClock()) },
				{ "powerConsumption", std::to_string(gpu_.getPowerConsumption()) },
				{ "powerLimit", std::to_string(gpu_.getPowerLimit()) },
				{ "streamingMultiprocessorClock", std::to_string(gpu_.getStreamingMultiprocessorClock()) },
				{ "temperature", std::to_string(gpu_.getTemperature()) },
			};
		}
	}
}