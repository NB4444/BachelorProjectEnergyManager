#pragma once

#include "EnergyManager/Hardware/Node.hpp"
#include "EnergyManager/Profiling/DeviceMonitor.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"

#include <memory>

namespace EnergyManager {
	namespace Profiling {
		class NodeMonitor : public DeviceMonitor {
			std::shared_ptr<Hardware::Node> node_;

			Utility::Units::Joule startEnergyConsumption_ = 0;

			bool startEnergyConsumptionMeasured_ = false;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			NodeMonitor(const std::shared_ptr<Hardware::Node>& node);
		};
	}
}