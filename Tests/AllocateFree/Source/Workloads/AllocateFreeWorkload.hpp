#pragma once

#include <EnergyManager/Benchmarking/Workloads/Workload.hpp>

namespace Workloads {
	/**
	 * A Workload that allocates a bunch of memory and then releases it.
	 */
	class AllocateFreeWorkload : public EnergyManager::Benchmarking::Workloads::Workload {
	public:
		/**
		 * Creates a new allocate free Workload.
		 * @param hostAllocations The amount of allocations to perform on the host.
		 * @param hostSize The size of each allocation on the host.
		 * @param deviceAllocations The amount of allocations to perform on the device.
		 * @param deviceSize The size of each allocation on the device.
		 */
		AllocateFreeWorkload(const unsigned int& hostAllocations, const size_t& hostSize, const unsigned int& deviceAllocations, const size_t& deviceSize);
	};
}
