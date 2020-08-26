#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Permillion : public PerUnit<double, unsigned long, double> {
			public:
				Permillion(const double& value = 0);
			};
		}
	}
}