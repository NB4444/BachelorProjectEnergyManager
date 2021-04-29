#include "./NodeMonitor.hpp"

#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> NodeMonitor::onPollDevice() {
				std::map<std::string, std::string> results = { { "memorySize", Utility::Text::toString(node_->getMemorySize().toValue()) },
															   { "freeMemorySize", Utility::Text::toString(node_->getFreeMemorySize().toValue()) },
															   { "usedMemorySize", Utility::Text::toString(node_->getUsedMemorySize().toValue()) },
															   { "sharedMemorySize", Utility::Text::toString(node_->getSharedMemorySize().toValue()) },
															   { "bufferMemorySize", Utility::Text::toString(node_->getBufferMemorySize().toValue()) },
															   { "swapMemorySize", Utility::Text::toString(node_->getSwapMemorySize().toValue()) },
															   { "freeSwapMemorySize", Utility::Text::toString(node_->getFreeSwapMemorySize().toValue()) },
															   { "usedSwapMemorySize", Utility::Text::toString(node_->getUsedSwapMemorySize().toValue()) },
															   { "highMemorySize", Utility::Text::toString(node_->getHighMemorySize().toValue()) },
															   { "freeHighMemorySize", Utility::Text::toString(node_->getFreeHighMemorySize().toValue()) },
															   { "usedHighMemorySize", Utility::Text::toString(node_->getUsedHighMemorySize().toValue()) },
															   { "processCount", Utility::Text::toString(node_->getProcessCount()) } };

				return results;
			}

			void NodeMonitor::onResetDevice() {
				startEnergyConsumption_ = 0;
				startEnergyConsumptionMeasured_ = false;
				
				node_->getEnergyConsumption();
			}

			NodeMonitor::NodeMonitor(const std::shared_ptr<Hardware::Node>& node, const std::chrono::system_clock::duration& interval) : DeviceMonitor("NodeMonitor", node, interval), node_(node) {
				reset();
			}
		}
	}
}