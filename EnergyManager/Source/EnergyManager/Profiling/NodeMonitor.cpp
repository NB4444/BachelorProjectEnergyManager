#include "./NodeMonitor.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> NodeMonitor::onPoll() {
			if(!startEnergyConsumptionMeasured_) {
				startEnergyConsumptionMeasured_ = true;
				startEnergyConsumption_ = cpu_.getEnergyConsumption() + gpu_.getEnergyConsumption();
			}

			return { { "energyConsumption", std::to_string(cpu_.getEnergyConsumption() + gpu_.getEnergyConsumption() - startEnergyConsumption_) },
					 { "powerConsumption", std::to_string(cpu_.getPowerConsumption() + gpu_.getPowerConsumption()) } };
		}

		NodeMonitor::NodeMonitor(const Hardware::CPU& cpu, const Hardware::GPU& gpu) : Monitor("NodeMonitor"), cpu_(cpu), gpu_(gpu) {
		}
	}
}