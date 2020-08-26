#include "./Watt.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Watt::Watt(const double& value, const SIPrefix& prefix) : SIUnit("Watt", "W", value, prefix) {
			}
		}
	}
}