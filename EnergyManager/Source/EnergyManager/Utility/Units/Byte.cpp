#include "./Byte.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Byte::Byte(const unsigned long& value, const SIPrefix& prefix) : SIUnit("Byte", "B", value, prefix) {
			}
		}
	}
}