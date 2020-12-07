#include "./Instruction.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Instruction::Instruction(const double& value, const SIPrefix& prefix) : SIUnit("Instruction", "I", value, prefix) {
			}
		}
	}
}