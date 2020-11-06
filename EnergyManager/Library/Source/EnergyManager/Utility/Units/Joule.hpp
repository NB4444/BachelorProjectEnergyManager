#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount of energy.
			 */
			class Joule : public SIUnit<Joule, double> {
			public:
				/**
				 * Creates a new Joule.
				 * @param value The amount of energy.
				 * @param prefix The SI prefix.
				 */
				Joule(const double& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}