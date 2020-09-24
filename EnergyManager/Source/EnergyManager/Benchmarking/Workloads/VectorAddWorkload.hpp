#pragma once

#include "SyntheticWorkload.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class VectorAddWorkload : public SyntheticWorkload {
			public:
				VectorAddWorkload(const size_t& size);
			};
		}
	}
}