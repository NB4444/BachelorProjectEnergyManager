#include "./Permillion.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Permillion::Permillion(const double& value) : PerUnit(value, 1e6l, "Permillion", "ppm") {
			}
		}
	}
}