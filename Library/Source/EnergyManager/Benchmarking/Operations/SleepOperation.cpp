#include "./SleepOperation.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			void SleepOperation::onRun() {
				sleep(duration_);
			}

			SleepOperation::SleepOperation(const std::chrono::system_clock::duration& duration) : duration_(duration) {
			}
		}
	}
}