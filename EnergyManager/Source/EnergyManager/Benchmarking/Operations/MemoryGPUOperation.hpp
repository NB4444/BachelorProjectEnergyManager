#pragma once

#include "EnergyManager/Benchmarking/Operations/SyntheticGPUOperation.hpp"

#include <vector>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class MemoryGPUOperation : public SyntheticGPUOperation {
			protected:
				static std::vector<std::pair<int*, size_t>> variables_;
			};
		}
	}
}