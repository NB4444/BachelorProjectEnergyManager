#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

#include <cstddef>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class AllocateGPUOperation : public MemoryGPUOperation {
				size_t size_;

			protected:
				void onRun() override;

			public:
				AllocateGPUOperation(const size_t& size = 8);
			};
		}
	}
}