#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Permyriad : public PerUnit<double, unsigned long, double> {
			public:
				Permyriad(const double& value = 0);
			};
		}
	}
}