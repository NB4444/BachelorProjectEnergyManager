#pragma once

#include "SyntheticWorkload.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class AllocateFreeWorkload : public SyntheticWorkload {
			public:
				AllocateFreeWorkload(const unsigned int& hostAllocations, const size_t& hostSize, const unsigned int& deviceAllocations, const size_t& deviceSize);
			};
		}
	}
}