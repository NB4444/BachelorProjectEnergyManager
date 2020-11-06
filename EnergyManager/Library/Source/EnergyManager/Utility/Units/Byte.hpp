#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount of memory.
			 */
			class Byte : public SIUnit<Byte, unsigned long> {
			public:
				/**
				 * Creates a new Byte.
				 * @param value The amount of bytes.
				 * @param prefix The SI prefix.
				 */
				Byte(const unsigned long& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}