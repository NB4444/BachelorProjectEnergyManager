#include "./DeviceMonitor.hpp"

#include "EnergyManager/Utility/Text.hpp"

#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> DeviceMonitor::onPoll() {
				if(!startEnergyConsumptionMeasured_) {
					startEnergyConsumptionMeasured_ = true;
					startEnergyConsumption_ = device_->getEnergyConsumption();
				}

				std::map<std::string, std::string> results = {
					{ "energyConsumption", Utility::Text::toString((device_->getEnergyConsumption() - startEnergyConsumption_).toValue()) },
					{ "powerConsumption", Utility::Text::toString(device_->getPowerConsumption().toValue()) },
				};

				// Get downstream values
				auto deviceResults = onPollDevice();
				results.insert(deviceResults.begin(), deviceResults.end());

				return results;
			}

			std::map<std::string, std::string> DeviceMonitor::onPollDevice() {
				return {};
			}

			void DeviceMonitor::onReset() {
				startEnergyConsumption_ = 0;
				startEnergyConsumptionMeasured_ = false;

				onResetDevice();
			}

			void DeviceMonitor::onResetDevice() {
			}

			DeviceMonitor::DeviceMonitor(const std::string& name, std::shared_ptr<Hardware::Device> device, const std::chrono::system_clock::duration& interval)
				: Monitor(name, interval)
				, device_(std::move(device)) {
				reset();
			}
		}
	}
}