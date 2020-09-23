#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryCPUOperation.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class FreeCPUOperation : public MemoryCPUOperation {
			protected:
				void onRun() override;

			public:
				FreeCPUOperation() = default;
			};
		}
	}
}