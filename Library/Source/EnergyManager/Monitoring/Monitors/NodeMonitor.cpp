#include "./NodeMonitor.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> NodeMonitor::onPollDevice() {
				std::map<std::string, std::string> results = { { "memorySize", std::to_string(node_->getMemorySize().toValue()) },
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

				return results;
			}

			void NodeMonitor::onResetDevice() {
				startEnergyConsumption_ = 0;
				startEnergyConsumptionMeasured_ = false;
			}

			NodeMonitor::NodeMonitor(const std::shared_ptr<Hardware::Node>& node, const std::chrono::system_clock::duration& interval) : DeviceMonitor("NodeMonitor", node, interval), node_(node) {
				reset();
			}
		}
	}
}