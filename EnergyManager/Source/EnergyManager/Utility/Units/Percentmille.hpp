#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Percentmille : public PerUnit<double, unsigned long, double> {
			public:
				Percentmille(const double& value = 0);
			};
		}
	}
}