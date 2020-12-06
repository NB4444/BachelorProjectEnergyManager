#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount per hundred.
			 */
			class Percent : public PerUnit<Percent, double, unsigned long, double> {
			public:
				/**
				 * Creates a new Percent.
				 * @param value The amount per hundred.
				 */
				Percent(const double& value = 0);
			};
		}
	}
}