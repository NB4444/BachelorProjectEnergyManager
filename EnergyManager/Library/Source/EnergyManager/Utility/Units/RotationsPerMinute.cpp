#include "./RotationsPerMinute.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			RotationsPerMinute::RotationsPerMinute(const Rotation& value) : PerUnit(value, 1, "Rotations Per Minute", "RPM") {
			}
		}
	}
}