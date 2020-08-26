#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Hertz :
				public SIUnit<Hertz, unsigned long> {
				public:
					Hertz(const unsigned long& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}