#pragma once

#include "SyntheticGPUWorkload.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class ActiveInactiveWorkload : public SyntheticGPUWorkload {
			public:
				ActiveInactiveWorkload(const unsigned int& activeOperations, const std::chrono::system_clock::duration& inactivePeriod, const unsigned int& cycles = 1);
			};
		}
	}
}