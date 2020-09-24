#include "./AllocateCPUOperation.hpp"

#include "EnergyManager/Hardware/CPU.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			void AllocateCPUOperation::onRun() {
				int* variable = (int*) malloc(size_);
				variables_.emplace_back(variable, size_);
			}

			AllocateCPUOperation::AllocateCPUOperation(const size_t& size) : size_(size) {
			}
		}
	}
}