#pragma once

#include "EnergyManager/Utility/Units/Joule.hpp"
#include "EnergyManager/Utility/Units/Watt.hpp"

namespace EnergyManager {
	namespace Hardware {
		/**
		 * Represents a hardware Device.
		 */
		class Device {
			public:
				/**
				 * Returns the Device's energy consumption since the time it was powered on.
				 * @return The energy consumption.
				 */
				virtual Utility::Units::Joule getEnergyConsumption() const = 0;

				/**
				 * Returns the Device's power consumption.
				 * @return The power consumption.
				 */
				virtual Utility::Units::Watt getPowerConsumption() const = 0;
		};
	}
}