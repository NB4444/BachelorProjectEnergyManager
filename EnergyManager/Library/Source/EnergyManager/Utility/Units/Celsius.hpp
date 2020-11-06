#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents a temperature.
			 */
			class Celsius : public SIUnit<Celsius, unsigned long> {
			public:
				/**
				 * Creates a new Celsius.
				 * @param value The temperature.
				 * @param prefix The SI prefix.
				 */
				Celsius(const unsigned long& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}