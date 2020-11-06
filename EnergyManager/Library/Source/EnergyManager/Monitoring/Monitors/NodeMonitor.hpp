#pragma once

#include "DeviceMonitor.hpp"
#include "EnergyManager/Hardware/Node.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"

#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a Node.
			 */
			class NodeMonitor : public DeviceMonitor {
				/**
				 * The Node to monitor.
				 */
				std::shared_ptr<Hardware::Node> node_;

				/**
				 * the energy consumption at the start of monitoring.
				 */
				Utility::Units::Joule startEnergyConsumption_;

				/**
				 * Whether the initial energy consumption has been measured yet.
				 */
				bool startEnergyConsumptionMeasured_;

			protected:
				std::map<std::string, std::string> onPollDevice() final;

				void onResetDevice() final;

			public:
				/**
				 * Creates a new NodeMonitor.
				 * @param node The Node to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				NodeMonitor(const std::shared_ptr<Hardware::Node>& node, const std::chrono::system_clock::duration& interval);
			};
		}
	}
}