#pragma once

#include "EnergyManager/Testing/Benchmarking/SyntheticGPUWorkload.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Benchmarking {
			class VectorAddWorkload : public SyntheticGPUWorkload {
			public:
				VectorAddWorkload(const size_t& size);
			};
		}
	}
}