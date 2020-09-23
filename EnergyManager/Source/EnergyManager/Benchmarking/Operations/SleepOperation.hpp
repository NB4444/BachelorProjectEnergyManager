#pragma once

#include "EnergyManager/Benchmarking/Operations/SyntheticOperation.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class SleepOperation : public SyntheticOperation {
				std::chrono::system_clock::duration duration_;

			protected:
				void onRun() override;

			public:
				SleepOperation(const std::chrono::system_clock::duration& duration);
			};
		}
	}
}