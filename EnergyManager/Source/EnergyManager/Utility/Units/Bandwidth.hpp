#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"
#include "EnergyManager/Utility/Units/Byte.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			class Bandwidth : public PerUnit<Byte, std::chrono::system_clock::duration, double> {
				public:
					Bandwidth(const Byte& value = 0, const std::chrono::system_clock::duration& duration = std::chrono::seconds(1));

					double toCombined() const override;
			};
		}
	}
}