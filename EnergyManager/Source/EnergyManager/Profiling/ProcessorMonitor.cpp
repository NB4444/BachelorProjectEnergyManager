#include "./ProcessorMonitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		ProcessorMonitor::ProcessorMonitor(const std::string& name, const std::shared_ptr<Hardware::Processor>& processor) : DeviceMonitor(name, processor), processor_(processor) {
		}

		std::map<std::string, std::string> ProcessorMonitor::onPoll() {
			auto processorResults = std::map<std::string, std::string> {
				{ "coreClockRate", std::to_string(processor_->getCoreClockRate().toValue()) },
				{ "coreUtilizationRate", std::to_string(processor_->getCoreUtilizationRate().toCombined()) },
				{ "maximumCoreClockRate", std::to_string(processor_->getMaximumCoreClockRate().toValue()) }
			};

			// Get upstream values
			auto deviceResults = DeviceMonitor::onPoll();
			processorResults.insert(deviceResults.begin(), deviceResults.end());

			return processorResults;
		}
	}
}