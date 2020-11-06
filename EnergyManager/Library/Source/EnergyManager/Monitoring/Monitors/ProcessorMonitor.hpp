#pragma once

#include "DeviceMonitor.hpp"
#include "EnergyManager/Hardware/Processor.hpp"

#include <string>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a Processor.
			 */
			class ProcessorMonitor : public DeviceMonitor {
				/**
				 * The Processor to monitor.
				 */
				std::shared_ptr<Hardware::Processor> processor_;

			protected:
				std::map<std::string, std::string> onPollDevice() final;

				virtual std::map<std::string, std::string> onPollProcessor();

			public:
				/**
				 * Creates a new ProcessorMonitor.
				 * @param name The name of the Monitor.
				 * @param processor The Processor to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				ProcessorMonitor(const std::string& name, const std::shared_ptr<Hardware::Processor>& processor, const std::chrono::system_clock::duration& interval);
			};
		}
	}
}