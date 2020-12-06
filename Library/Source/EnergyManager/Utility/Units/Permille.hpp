#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount per thousand.
			 */
			class Permille : public PerUnit<Permille, double, unsigned long, double> {
			public:
				/**
				 * Creates a new Permille.
				 * @param value The amount per thousand.
				 */
				Permille(const double& value = 0);
			};
		}
	}
}