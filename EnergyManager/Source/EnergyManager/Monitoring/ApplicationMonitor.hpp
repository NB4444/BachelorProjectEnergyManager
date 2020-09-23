#pragma once

#include "EnergyManager/Application.hpp"
#include "EnergyManager/Monitoring/Monitor.hpp"

#include <map>

namespace EnergyManager {
	namespace Monitoring {
		class ApplicationMonitor : public Monitor {
			const Application& application_;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			ApplicationMonitor(const Application& application);
		};
	}
}