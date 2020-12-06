#pragma once

#include "EnergyManager/Hardware/Device.hpp"
#include "EnergyManager/Utility/Units/Celsius.hpp"
#include "EnergyManager/Utility/Units/Hertz.hpp"
#include "EnergyManager/Utility/Units/Percent.hpp"

namespace EnergyManager {
	namespace Hardware {
		class Processor : public Device {
			/**
			 * The ID of the device.
			 */
			unsigned int id_;

		protected:
			/**
			 * Creates a new Processor.
			 * @param id The ID of the device.
			 */
			explicit Processor(const unsigned int& id);

		public:
			/**
			 * Gets the ID of the CentralProcessor.
			 * @return The ID.
			 */
			unsigned int getID() const;

			/**
			 * Gets the core clock rate.
			 * @return The core clock rate.
			 */
			virtual Utility::Units::Hertz getCoreClockRate() const = 0;

			/**
			 * Gets the current minimum core clock rate.
			 * @return The minimum core clock rate.
			 */
			virtual Utility::Units::Hertz getCurrentMinimumCoreClockRate() const = 0;

			/**
			 * Gets the current maximum core clock rate.
			 * @return The maximum core clock rate.
			 */
			virtual Utility::Units::Hertz getCurrentMaximumCoreClockRate() const = 0;

			/**
			 * Sets the clock rate.
			 * @param rate The clock rate.
			 */
			void setCoreClockRate(const Utility::Units::Hertz& rate);

			/**
			 * Sets the clock rate boundaries.
			 * @param minimumRate The minimum clock rate.
			 * @param maximumRate The maximum clock rate.
			 */
			virtual void setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) = 0;

			/**
			 * Resets the clock rate to the default value.
			 */
			virtual void resetCoreClockRate() = 0;

			/**
			 * Gets the core utilization rate.
			 * Percent of time over the past sample period during which the core was active.
			 * @return The core utilization rate.
			 */
			virtual Utility::Units::Percent getCoreUtilizationRate() = 0;

			/**
			 * Gets the minimum supported clock rate.
			 * @return The minimum supported clock rate.
			 */
			virtual Utility::Units::Hertz getMinimumCoreClockRate() const = 0;

			/**
			 * Retrieves the maximum clock speeds for the device.
			 * @return The maximum core clock rate.
			 */
			virtual Utility::Units::Hertz getMaximumCoreClockRate() const = 0;

			/**
			 * Gets the temperature of the device.
			 * @return The temperature.
			 */
			virtual Utility::Units::Celsius getTemperature() const = 0;
		};
	}
}