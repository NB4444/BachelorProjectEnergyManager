#include "./Hertz.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Hertz::Hertz(const unsigned long& value, const SIPrefix& prefix) : SIUnit("Hertz", "Hz", value, prefix) {
			}
		}
	}
}