#pragma once

#include "SyntheticWorkload.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class ActiveInactiveWorkload : public SyntheticWorkload {
			public:
				static void initialize();

				ActiveInactiveWorkload(const unsigned int& activeOperations, const std::chrono::system_clock::duration& inactivePeriod, const unsigned int& cycles = 1);
			};
		}
	}
}