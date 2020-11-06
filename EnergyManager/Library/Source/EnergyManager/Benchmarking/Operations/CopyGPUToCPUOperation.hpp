#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

#include <cstddef>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * A copy Operation that copies variables from the GPU to the CPU.
			 */
			class CopyGPUToCPUOperation
				: public MemoryCPUOperation
				, public MemoryGPUOperation {
				/**
				 * The amount of variables to copy.
				 */
				unsigned int count_;

			protected:
				void onRun() override;

			public:
				/**
				 * Creates a new copy Operation that copies variables from the GPU to the CPU.
				 * @param count The amount of variables to copy.
				 */
				explicit CopyGPUToCPUOperation(const unsigned int& count = 1);
			};
		}
	}
}