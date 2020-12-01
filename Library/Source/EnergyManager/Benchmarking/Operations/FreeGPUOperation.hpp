#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * A free Operation that runs on the GPU.
			 */
			class FreeGPUOperation : public MemoryGPUOperation {
			protected:
				void onRun() final;

			public:
				/**
				 * Creates a new free Operation that runs on the GPU.
				 */
				FreeGPUOperation() = default;
			};
		}
	}
}