#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"
#include "EnergyManager/Utility/Units/Rotation.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class RotationsPerMinute :
				public PerUnit<Rotation, int, double> {
				public:
					RotationsPerMinute(const Rotation& value = {});
			};
		}
	}
}