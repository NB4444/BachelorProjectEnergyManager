#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

#include <cstddef>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * A copy Operation that copies variables from the CPU to the GPU.
			 */
			class CopyCPUToGPUOperation
				: public MemoryCPUOperation
				, public MemoryGPUOperation {
				/**
				 * The amount of variables to copy.
				 */
				unsigned int count_;

			protected:
				void onRun() final;

			public:
				/**
				 * Creates a new copy Operation that copies variables from the CPU to the GPU.
				 * @param count The amount of variables to copy.
				 */
				explicit CopyCPUToGPUOperation(const unsigned int& count = 1);
			};
		}
	}
}