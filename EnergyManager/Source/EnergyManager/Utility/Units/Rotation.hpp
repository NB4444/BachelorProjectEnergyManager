#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Rotation : public SIUnit<Rotation, double> {
				public:
					Rotation(const double& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}