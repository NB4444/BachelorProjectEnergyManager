#include "./DeviceMonitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		DeviceMonitor::DeviceMonitor(const std::string& name, const Hardware::Device& device) : Monitor(name), device_(device) {
		}

		std::map<std::string, std::string> DeviceMonitor::onPoll() {
			if(!startEnergyConsumptionMeasured_) {
				startEnergyConsumptionMeasured_ = true;
				startEnergyConsumption_ = device_.getEnergyConsumption();
			}

			return {
				{ "energyConsumption", std::to_string(device_.getEnergyConsumption() - startEnergyConsumption_) },
				{ "powerConsumption", std::to_string(device_.getPowerConsumption()) },
			};
		}
	}
}