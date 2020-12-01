#include "./Percent.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Percent::Percent(const double& value) : PerUnit(value, 1e2l, "Percent", "%") {
			}
		}
	}
}