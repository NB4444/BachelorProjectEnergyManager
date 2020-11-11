#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

#include <cstddef>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * An allocate Operation that runs on the GPU.
			 */
			class AllocateGPUOperation : public MemoryGPUOperation {
				/**
				 * The amount of memory to allocate.
				 */
				size_t size_;

			protected:
				void onRun() final;

			public:
				/**
				 * Creates a new allocate Operation that runs on the GPU.
				 * @param size The amount of memory to allocate.
				 */
				explicit AllocateGPUOperation(const size_t& size = 8);
			};
		}
	}
}