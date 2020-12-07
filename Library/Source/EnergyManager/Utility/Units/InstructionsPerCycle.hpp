#pragma once

#include "EnergyManager/Utility/Units/PerUnit.hpp"
#include "EnergyManager/Utility/Units/Instruction.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents an amount of Instructions per minute.
			 */
			class InstructionsPerCycle : public PerUnit<InstructionsPerCycle, Instruction, int, double> {
			public:
				/**
				 * Creates a new InstructionsPerCycle.
				 * @param value The amount of Instructions per minute.
				 */
				InstructionsPerCycle(const Instruction& value = {});
			};
		}
	}
}