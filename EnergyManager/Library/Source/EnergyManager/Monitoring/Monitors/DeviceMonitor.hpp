#pragma once

#include "EnergyManager/Hardware/Device.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"
#include "Monitor.hpp"

#include <memory>
#include <string>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a Device.
			 */
			class DeviceMonitor : public Monitor {
				/**
				 * The Device to monitor.
				 */
				std::shared_ptr<Hardware::Device> device_;

				/**
				 * the energy consumption at the start of monitoring.
				 */
				Utility::Units::Joule startEnergyConsumption_;

				/**
				 * Whether the initial energy consumption has been measured yet.
				 */
				bool startEnergyConsumptionMeasured_;

			protected:
				std::map<std::string, std::string> onPoll() final;

				virtual std::map<std::string, std::string> onPollDevice();

				void onReset() final;

				virtual void onResetDevice();

			public:
				/**
				 * Creates a new DeviceMonitor.
				 * @param name The name of the Monitor.
				 * @param device The Device to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				DeviceMonitor(const std::string& name, std::shared_ptr<Hardware::Device> device, const std::chrono::system_clock::duration& interval);
			};
		}
	}
}