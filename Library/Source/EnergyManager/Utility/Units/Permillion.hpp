#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount per million.
			 */
			class Permillion : public PerUnit<double, unsigned long, double> {
			public:
				/**
				 * Creates a new Permillion.
				 * @param value The amount per million.
				 */
				Permillion(const double& value = 0);
			};
		}
	}
}