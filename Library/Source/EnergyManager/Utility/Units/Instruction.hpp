#pragma once

#include "EnergyManager/Utility/Units/SIUnit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount of Instructions.
			 */
			class Instruction : public SIUnit<Instruction, double> {
			public:
				/**
				 * Creates a new Instruction.
				 * @param value The amount of Instructions.
				 * @param prefix The SI prefix.
				 */
				Instruction(const double& value = 0, const SIPrefix& prefix = SIPrefix::NONE);
			};
		}
	}
}