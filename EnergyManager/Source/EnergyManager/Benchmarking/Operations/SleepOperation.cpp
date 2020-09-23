#include "./SleepOperation.hpp"

#include <unistd.h>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			void SleepOperation::onRun() {
				usleep(std::chrono::duration_cast<std::chrono::microseconds>(duration_).count());
			}

			SleepOperation::SleepOperation(const std::chrono::system_clock::duration& duration) : duration_(duration) {
			}
		}
	}
}