#include "./Rotation.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Rotation::Rotation(const double& value, const SIPrefix& prefix) : SIUnit("Rotation", "R", value, prefix) {
			}
		}
	}
}