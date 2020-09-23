#pragma once

#include "SyntheticGPUWorkload.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class AllocateFreeWorkload : public SyntheticGPUWorkload {
			public:
				AllocateFreeWorkload(const unsigned int& hostAllocations, const size_t& hostSize, const unsigned int& deviceAllocations, const size_t& deviceSize);
			};
		}
	}
}