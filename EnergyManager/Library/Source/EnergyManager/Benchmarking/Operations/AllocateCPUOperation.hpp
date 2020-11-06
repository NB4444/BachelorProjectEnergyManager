#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryCPUOperation.hpp"

#include <cstddef>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * An allocate Operation that runs on the CPU.
			 */
			class AllocateCPUOperation : public MemoryCPUOperation {
				/**
				 * The amount of memory to allocate.
				 */
				size_t size_;

			protected:
				void onRun() override;

			public:
				/**
				 * Creates a new allocate operation that runs on the CPU.
				 * @param size The amount of memory to allocate.
				 */
				explicit AllocateCPUOperation(const size_t& size = 8);
			};
		}
	}
}