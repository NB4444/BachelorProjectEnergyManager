#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Watt : public SIUnit<Watt, double> {
			public:
				Watt(const double& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}