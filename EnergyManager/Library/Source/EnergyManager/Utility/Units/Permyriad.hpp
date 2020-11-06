#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount per ten thousand.
			 */
			class Permyriad : public PerUnit<double, unsigned long, double> {
			public:
				/**
				 * Creates a new Permyriad.
				 * @param value The amount per ten thousand.
				 */
				Permyriad(const double& value = 0);
			};
		}
	}
}