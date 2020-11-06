#pragma once

#include <EnergyManager/Benchmarking/Workloads/Workload.hpp>
#include <chrono>

namespace Workloads {
	/**
	 * A Workload that does a set amount of iterations where it spends some time calculating followed by a period of sleep.
	 */
	class ActiveInactiveWorkload : public EnergyManager::Benchmarking::Workloads::Workload {
	public:
		/**
		 * Creates a new active inactive Workload.
		 * @param activeOperations The amount of active Operations to perform.
		 * @param inactivePeriod The amount of time to spend idle.
		 * @param cycles The amount of active inactive cycles to perform.
		 */
		ActiveInactiveWorkload(const unsigned int& activeOperations, const std::chrono::system_clock::duration& inactivePeriod, const unsigned int& cycles = 1);
	};
}