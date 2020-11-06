#pragma once

#include "EnergyManager/Testing/Application.hpp"
#include "Monitor.hpp"

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
				const Testing::Application& application_;

			protected:
				std::map<std::string, std::string> onPoll() final;

			public:
				/**
				 * Creates a new ApplicationMonitor.
				 * @param application The Application to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				explicit ApplicationMonitor(const Testing::Application& application, const std::chrono::system_clock::duration& interval);
			};
		}
	}
}