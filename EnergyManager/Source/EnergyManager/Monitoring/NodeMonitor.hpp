#pragma once

#include "EnergyManager/Hardware/Node.hpp"
#include "EnergyManager/Monitoring/DeviceMonitor.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"

#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		class NodeMonitor : public DeviceMonitor {
			std::shared_ptr<Hardware::Node> node_;

			Utility::Units::Joule startEnergyConsumption_ = 0;

			bool startEnergyConsumptionMeasured_ = false;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			static void initialize();

			NodeMonitor(const std::shared_ptr<Hardware::Node>& node);
		};
	}
}