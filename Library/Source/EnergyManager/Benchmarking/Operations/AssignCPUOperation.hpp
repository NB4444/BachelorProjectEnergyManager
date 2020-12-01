#pragma once

#include "EnergyManager/Benchmarking/Operations/MemoryCPUOperation.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * An assign Operation that runs on the CPU.
			 */
			class AssignCPUOperation : public MemoryCPUOperation {
				/**
				 * The amount of variables to assign values to.
				 */
				unsigned int count_;

			protected:
				void onRun() final;

			public:
				/**
				 * Creates a new assign Operation that runs on the CPU.
				 * @param count The amount of variables to assign values to.
				 */
				explicit AssignCPUOperation(const unsigned int& count = 1);
			};
		}
	}
}