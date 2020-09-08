#include "./NodeMonitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> NodeMonitor::onPoll() {
			std::map<std::string, std::string> nodeResults = { { "memorySize", std::to_string(node_->getMemorySize().toValue()) },
															   { "freeMemorySize", std::to_string(node_->getFreeMemorySize().toValue()) },
															   { "usedMemorySize", std::to_string(node_->getUsedMemorySize().toValue()) },
															   { "sharedMemorySize", std::to_string(node_->getSharedMemorySize().toValue()) },
															   { "bufferMemorySize", std::to_string(node_->getBufferMemorySize().toValue()) },
															   { "swapMemorySize", std::to_string(node_->getSwapMemorySize().toValue()) },
															   { "freeSwapMemorySize", std::to_string(node_->getFreeSwapMemorySize().toValue()) },
															   { "usedSwapMemorySize", std::to_string(node_->getUsedSwapMemorySize().toValue()) },
															   { "highMemorySize", std::to_string(node_->getHighMemorySize().toValue()) },
															   { "freeHighMemorySize", std::to_string(node_->getFreeHighMemorySize().toValue()) },
															   { "usedHighMemorySize", std::to_string(node_->getUsedHighMemorySize().toValue()) },
															   { "processCount", std::to_string(node_->getProcessCount()) } };

			// Get upstream values
			auto deviceResults = DeviceMonitor::onPoll();
			nodeResults.insert(deviceResults.begin(), deviceResults.end());

			return nodeResults;
		}

		NodeMonitor::NodeMonitor(const std::shared_ptr<Hardware::Node>& node) : DeviceMonitor("NodeMonitor", node), node_(node) {
		}
	}
}