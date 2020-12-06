#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount per hundred thousand.
			 */
			class Percentmille : public PerUnit<Percentmille, double, unsigned long, double> {
			public:
				/**
				 * Creates a new Percentmille.
				 * @param value The amount per hundred thousand.
				 */
				Percentmille(const double& value = 0);
			};
		}
	}
}