#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

#include <cstddef>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class CopyGPUToCPUOperation
				: public MemoryCPUOperation
				, public MemoryGPUOperation {
				unsigned int count_;

			protected:
				void onRun() override;

			public:
				CopyGPUToCPUOperation(const unsigned int& count = 1);
			};
		}
	}
}