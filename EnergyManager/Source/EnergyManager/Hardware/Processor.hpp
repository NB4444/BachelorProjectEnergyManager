#pragma once

#include "EnergyManager/Hardware/Device.hpp"

namespace EnergyManager {
	namespace Hardware {
		class Processor :
			public Device {
			public:
				/**
				 * Gets the core clock rate.
				 * @return The core clock rate in Hertz.
				 */
				virtual unsigned long getCoreClockRate() const = 0;

				/**
				 * Sets the core clock rate.
				 * @param rate The core clock rate in Hertz.
				 */
				virtual void setCoreClockRate(unsigned long& rate) = 0;

				/**
				 * Gets the core utilization rate.
				 * Percent of time over the past sample period during which the core was active.
				 * @return The core utilization rate.
				 */
				virtual float getCoreUtilizationRate() const = 0;

				/**
				 * Retrieves the maximum clock speeds for the device.
				 * @return The maximum core clock rate in Hertz.
				 */
				virtual unsigned long getMaximumCoreClockRate() const = 0;
		};
	}
}