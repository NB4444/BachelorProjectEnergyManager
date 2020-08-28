#pragma once

#include "EnergyManager/Testing/Benchmarking/SyntheticGPUWorkload.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Benchmarking {
			class AllocateFreeWorkload : public SyntheticGPUWorkload {
			public:
				AllocateFreeWorkload(const unsigned int& hostAllocations, const size_t& hostSize, const unsigned int& deviceAllocations, const size_t& deviceSize);
			};
		}
	}
}