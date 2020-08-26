#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Celsius : public SIUnit<Celsius, unsigned long> {
			public:
				Celsius(const unsigned long& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}