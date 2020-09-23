#pragma once

#include "EnergyManager/Hardware/Device.hpp"
#include "EnergyManager/Monitoring/Monitor.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"

#include <memory>
#include <string>

namespace EnergyManager {
	namespace Monitoring {
		class DeviceMonitor : public Monitor {
			std::shared_ptr<Hardware::Device> device_;

			Utility::Units::Joule startEnergyConsumption_ = 0;

			bool startEnergyConsumptionMeasured_ = false;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			DeviceMonitor(const std::string& name, const std::shared_ptr<Hardware::Device>& device);
		};
	}
}