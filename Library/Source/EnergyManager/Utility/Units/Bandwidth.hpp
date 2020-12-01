#pragma once

#include "EnergyManager/Utility/Units/Byte.hpp"
#include "EnergyManager/Utility/Units/PerUnit.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents a transfer speed and amount.
			 */
			class Bandwidth : public PerUnit<Byte, std::chrono::system_clock::duration, double> {
			public:
				/**
				 * Creates a new Bandwidth.
				 * @param value The amount of bytes.
				 * @param duration The timespan in which the amount of bytes gets transferred.
				 */
				Bandwidth(const Byte& value = 0, const std::chrono::system_clock::duration& duration = std::chrono::seconds(1));

				double toCombined() const final;
			};
		}
	}
}