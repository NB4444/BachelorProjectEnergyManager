#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Joule : public SIUnit<Joule, double> {
				public:
					Joule(const double& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}