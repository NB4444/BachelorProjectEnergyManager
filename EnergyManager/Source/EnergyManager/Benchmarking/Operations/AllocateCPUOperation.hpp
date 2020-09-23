#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryCPUOperation.hpp"

#include <cstddef>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class AllocateCPUOperation : public MemoryCPUOperation {
				size_t size_;

			protected:
				void onRun() override;

			public:
				AllocateCPUOperation(const size_t& size = 8);
			};
		}
	}
}