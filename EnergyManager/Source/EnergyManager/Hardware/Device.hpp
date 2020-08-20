#pragma once

namespace EnergyManager {
	namespace Hardware {
		/**
		 * Represents a hardware Device.
		 */
		class Device {
		public:
			/**
			 * Returns the Device's energy consumption since the time it was powered on.
			 * @return The energy consumption in Joules.
			 */
			virtual float getEnergyConsumption() const = 0;

			/**
			 * Returns the Device's power consumption.
			 * @return The power consumption in Watts.
			 */
			virtual float getPowerConsumption() const = 0;
		};
	}
}