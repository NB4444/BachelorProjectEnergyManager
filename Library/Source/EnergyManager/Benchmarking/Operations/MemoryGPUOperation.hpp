#pragma once

#include "EnergyManager/Benchmarking/Operations/GPUOperation.hpp"

#include <vector>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * A memory Operation that runs on the GPU.
			 */
			class MemoryGPUOperation : public GPUOperation {
			protected:
				/**
				 * The variables that are stored on the GPU.
				 */
				static std::vector<std::pair<int*, size_t>> variables_;
			};
		}
	}
}