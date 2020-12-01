#include "./Joule.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Joule::Joule(const double& value, const SIPrefix& prefix) : SIUnit("Joule", "J", value, prefix) {
			}
		}
	}
}