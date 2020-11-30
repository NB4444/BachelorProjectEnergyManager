#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents a frequency.
			 */
			class Hertz : public SIUnit<Hertz, double> {
			public:
				/**
				 * Creates a new Hertz.
				 * @param value The frequency.
				 * @param prefix The SI prefix.
				 */
				Hertz(const double& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}