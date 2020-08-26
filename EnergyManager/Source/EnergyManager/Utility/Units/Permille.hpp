#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Permille : public PerUnit<double, unsigned long, double> {
			public:
				Permille(const double& value = 0);
			};
		}
	}
}