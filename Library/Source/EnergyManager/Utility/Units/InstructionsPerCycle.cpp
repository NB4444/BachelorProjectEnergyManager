#include "./InstructionsPerCycle.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			InstructionsPerCycle::InstructionsPerCycle(const Instruction& value) : PerUnit(value, 1, "Instructions Per Cycle", "IPC") {
			}
		}
	}
}