#pragma once

#include "EnergyManager/Hardware/Processor.hpp"
#include "EnergyManager/Profiling/DeviceMonitor.hpp"

#include <string>

namespace EnergyManager {
	namespace Profiling {
		class ProcessorMonitor :
			public DeviceMonitor {
				std::shared_ptr<Hardware::Processor> processor_;

			public:
				ProcessorMonitor(const std::string& name, const std::shared_ptr<Hardware::Processor>& processor);

			protected:
				std::map<std::string, std::string> onPoll() override;
		};
	}
}