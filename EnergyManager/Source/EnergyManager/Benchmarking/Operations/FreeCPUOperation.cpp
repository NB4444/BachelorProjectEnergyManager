#include "./FreeCPUOperation.hpp"

#include "EnergyManager/Hardware/CPU.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			void FreeCPUOperation::onRun() {
				int* variable = variables_.back().first;
				variables_.pop_back();
				free(variable);
			}
		}
	}
}