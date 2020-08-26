#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Percent : public PerUnit<double, unsigned long, double> {
			public:
				Percent(const double& value = 0);
			};
		}
	}
}