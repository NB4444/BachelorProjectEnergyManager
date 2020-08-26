#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Byte :
				public SIUnit<Byte, unsigned long> {
				public:
					Byte(const unsigned long& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}