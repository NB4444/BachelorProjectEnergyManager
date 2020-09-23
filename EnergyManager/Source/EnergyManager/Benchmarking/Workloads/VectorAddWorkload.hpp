#pragma once

#include "SyntheticGPUWorkload.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class VectorAddWorkload : public SyntheticGPUWorkload {
			public:
				VectorAddWorkload(const size_t& size);
			};
		}
	}
}