#include "./Permyriad.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Permyriad::Permyriad(const double& value) : PerUnit(value, 1e4l, "Permyriad", "‱") {
			}
		}
	}
}