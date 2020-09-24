#pragma once

#include "EnergyManager/Monitoring/Monitor.hpp"
#include "EnergyManager/Testing/Application.hpp"

#include <map>

namespace EnergyManager {
	namespace Monitoring {
		class ApplicationMonitor : public Monitor {
			const Testing::Application& application_;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			ApplicationMonitor(const Testing::Application& application);
		};
	}
}