#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Profiling/Monitor.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"

#include <memory>

namespace EnergyManager {
	namespace Profiling {
		class NodeMonitor :
			public Monitor {
				std::shared_ptr<Hardware::CPU> cpu_;

				std::shared_ptr<Hardware::GPU> gpu_;

				Utility::Units::Joule startEnergyConsumption_ = 0;

				bool startEnergyConsumptionMeasured_ = false;

			protected:
				std::map<std::string, std::string> onPoll() override;

			public:
				NodeMonitor(const std::shared_ptr<Hardware::CPU>& cpu, const std::shared_ptr<Hardware::GPU>& gpu);
		};
	}
}