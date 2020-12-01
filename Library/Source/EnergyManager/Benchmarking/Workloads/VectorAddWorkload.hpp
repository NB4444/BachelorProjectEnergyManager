#pragma once

#include "EnergyManager/Benchmarking/Workloads/Workload.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			/**
			 * A Workload that adds two vectors of a specified size.
			 */
			class VectorAddWorkload : public EnergyManager::Benchmarking::Workloads::Workload {
			public:
				/**
				 * Creates a new vector add Workload.
				 * @param size The size of the vectors to add.
				 */
				explicit VectorAddWorkload(const size_t& size);
			};
		}
	}
}