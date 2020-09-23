#pragma once

#include "EnergyManager/Benchmarking/Operations/SyntheticCPUOperation.hpp"

#include <vector>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class MemoryCPUOperation : public SyntheticCPUOperation {
			protected:
				static std::vector<std::pair<int*, size_t>> variables_;
			};
		}
	}
}