#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryGPUOperation.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class FreeGPUOperation : public MemoryGPUOperation {
			protected:
				void onRun() override;

			public:
				FreeGPUOperation() = default;
			};
		}
	}
}