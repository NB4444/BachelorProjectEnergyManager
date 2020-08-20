#pragma once

#include "EnergyManager/Hardware/Device.hpp"
#include "EnergyManager/Profiling/Monitor.hpp"

#include <string>

namespace EnergyManager {
	namespace Profiling {
		class DeviceMonitor : public Monitor {
			const Hardware::Device& device_;

			float startEnergyConsumption_ = 0;

			bool startEnergyConsumptionMeasured_ = false;

		public:
			DeviceMonitor(const std::string& name, const Hardware::Device& device);

		protected:
			std::map<std::string, std::string> onPoll() override;
		};
	}
}