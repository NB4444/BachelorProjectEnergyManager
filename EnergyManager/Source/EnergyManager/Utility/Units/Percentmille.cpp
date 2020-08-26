#include "./Percentmille.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Percentmille::Percentmille(const double& value) : PerUnit(value, 1e5l, "Percentmille", "pcm") {
			}
		}
	}
}