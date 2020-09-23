#include "./AssignCPUOperation.hpp"

#include "EnergyManager/Hardware/CPU.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			void AssignCPUOperation::onRun() {
				auto count = count_;

				size_t variableIndex = variables_.size() - 1;
				while(count > 0 && variableIndex > 0) {
					int* hostVariable = variables_[variableIndex].first;
					--variableIndex;

					*hostVariable = std::rand();

					--count;
				}
			}

			AssignCPUOperation::AssignCPUOperation(const unsigned int& count) : count_(count) {
			}
		}
	}
}