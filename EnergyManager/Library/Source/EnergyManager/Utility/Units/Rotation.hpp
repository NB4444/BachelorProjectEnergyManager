#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount of rotations.
			 */
			class Rotation : public SIUnit<Rotation, double> {
			public:
				/**
				 * Creates a new Rotation.
				 * @param value The amount of rotations.
				 * @param prefix The SI prefix.
				 */
				Rotation(const double& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}