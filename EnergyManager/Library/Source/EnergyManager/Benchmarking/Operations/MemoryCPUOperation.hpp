#pragma once

#include "EnergyManager/Benchmarking/Operations/CPUOperation.hpp"

#include <vector>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * A memory Operation that runs on the CPU.
			 */
			class MemoryCPUOperation : public CPUOperation {
			protected:
				/**
				 * The variables that are stored on the CPU.
				 */
				static std::vector<std::pair<int*, size_t>> variables_;
			};
		}
	}
}