#pragma once

#include "SyntheticWorkload.hpp"

#include <cuda_runtime.h>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class SyntheticGPUWorkload : public SyntheticWorkload {
			public:
				using SyntheticWorkload::SyntheticWorkload;
			};
		}
	}
}