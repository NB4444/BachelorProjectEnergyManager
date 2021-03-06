#pragma once

#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"
#include "EnergyManager/Utility//Application.hpp"

#include <map>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a specific Application.
			 */
			class ApplicationMonitor : public Monitor {
				/**
				 * The Application to monitor.
				 */
				const Utility::Application& application_;

			protected:
				std::map<std::string, std::string> onPoll() final;

			public:
				/**
				 * Creates a new ApplicationMonitor.
				 * @param application The Application to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				explicit ApplicationMonitor(const Utility::Application& application, const std::chrono::system_clock::duration& interval);
			};
		}
	}
}