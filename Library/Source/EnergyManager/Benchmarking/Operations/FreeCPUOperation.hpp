#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryCPUOperation.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * A free Operation that runs on the CPU.
			 */
			class FreeCPUOperation : public MemoryCPUOperation {
			protected:
				void onRun() final;

			public:
				/**
				 * Creates a new free Operation that runs on the CPU.
				 */
				FreeCPUOperation() = default;
			};
		}
	}
}