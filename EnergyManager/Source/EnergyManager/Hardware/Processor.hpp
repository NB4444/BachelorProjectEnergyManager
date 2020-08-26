#pragma once

#include "EnergyManager/Hardware/Device.hpp"
#include "EnergyManager/Utility/Units/Hertz.hpp"
#include "EnergyManager/Utility/Units/Percent.hpp"

namespace EnergyManager {
	namespace Hardware {
		class Processor : public Device {
		public:
			/**
			 * Gets the core clock rate.
			 * @return The core clock rate.
			 */
			virtual Utility::Units::Hertz getCoreClockRate() const = 0;

			/**
			 * Sets the core clock rate.
			 * @param rate The core clock rate.
			 */
			void setCoreClockRate(const Utility::Units::Hertz& rate);

			virtual void setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) = 0;

			virtual void resetCoreClockRate() = 0;

			/**
			 * Gets the core utilization rate.
			 * Percent of time over the past sample period during which the core was active.
			 * @return The core utilization rate.
			 */
			virtual Utility::Units::Percent getCoreUtilizationRate() const = 0;

			/**
			 * Retrieves the maximum clock speeds for the device.
			 * @return The maximum core clock rate.
			 */
			virtual Utility::Units::Hertz getMaximumCoreClockRate() const = 0;
		};
	}
}