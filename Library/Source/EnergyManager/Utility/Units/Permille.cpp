#include "./Permille.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Permille::Permille(const double& value) : PerUnit(value, 1e3l, "Permille", "‰") {
			}
		}
	}
}