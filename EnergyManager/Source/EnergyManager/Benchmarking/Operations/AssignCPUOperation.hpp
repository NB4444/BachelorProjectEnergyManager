#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryCPUOperation.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class AssignCPUOperation : public MemoryCPUOperation {
				unsigned int count_;

			protected:
				void onRun() override;

			public:
				AssignCPUOperation(const unsigned int& count = 1);
			};
		}
	}
}