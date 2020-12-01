#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount of Joules per second.
			 */
			class Watt : public SIUnit<Watt, double> {
			public:
				/**
				 * Creates a new Watt.
				 * @param value The amount of Joules per second.
				 * @param prefix The SI prefix.
				 */
				Watt(const double& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}