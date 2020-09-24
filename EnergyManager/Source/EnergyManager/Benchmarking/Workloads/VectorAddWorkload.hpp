#pragma once

#include "SyntheticWorkload.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class VectorAddWorkload : public SyntheticWorkload {
			public:
				static void initialize();

				VectorAddWorkload(const size_t& size);
			};
		}
	}
}