#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"
#include "EnergyManager/Utility/Units/Rotation.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount of rotations per minute.
			 */
			class RotationsPerMinute : public PerUnit<RotationsPerMinute, Rotation, int, double> {
			public:
				/**
				 * Creates a new RotationsPerMinute.
				 * @param value The amount of rotations per minute.
				 */
				RotationsPerMinute(const Rotation& value = {});
			};
		}
	}
}