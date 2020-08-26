#include "./Celsius.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Celsius::Celsius(const unsigned long& value, const SIPrefix& prefix) : SIUnit("Celsius", "C", value, prefix) {
			}
		}
	}
}