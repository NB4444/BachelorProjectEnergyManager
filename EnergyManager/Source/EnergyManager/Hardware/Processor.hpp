#pragma once

#include "EnergyManager/Hardware/Device.hpp"

namespace EnergyManager {
	namespace Hardware {
		class Processor : public Device {
		public:
			/**
			 * Gets the core clock rate.
			 * @return The core clock rate.
			 */
			virtual unsigned long getCoreClockRate() const = 0;

			/**
			 * Retrieves the maximum clock speeds for the device.
			 * @return The maximum core clock rate.
			 */
			virtual unsigned long getMaximumCoreClockRate() const = 0;
		};
	}
}