#pragma once

#include "EnergyManager/Benchmarking/Operations/Operation.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * A sleep Operation that pauses execution.
			 */
			class SleepOperation : public Operation {
				/**
				 * The amount of time to sleep.
				 */
				std::chrono::system_clock::duration duration_;

			protected:
				void onRun() final;

			public:
				/**
				 * Creates a new sleep Operation.
				 * @param duration The amount of time to sleep.
				 */
				explicit SleepOperation(const std::chrono::system_clock::duration& duration);
			};
		}
	}
}