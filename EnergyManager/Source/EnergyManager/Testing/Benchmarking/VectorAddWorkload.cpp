#include "./VectorAddWorkload.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Benchmarking {
			VectorAddWorkload::VectorAddWorkload(const size_t& size) {
				auto sizeString = std::to_string(size);

				// Allocate host vectors
				addOperation(SyntheticGPUOperation::HOST_ALLOCATE, { { "size", sizeString } });
				addOperation(SyntheticGPUOperation::HOST_ALLOCATE, { { "size", sizeString } });
				addOperation(SyntheticGPUOperation::HOST_ALLOCATE, { { "size", sizeString } });

				// Assign values
				addOperation(SyntheticGPUOperation::HOST_ASSIGN, { { "count", "3" } });

				// Allocate device vectors
				addOperation(SyntheticGPUOperation::DEVICE_ALLOCATE, { { "size", sizeString } });
				addOperation(SyntheticGPUOperation::DEVICE_ALLOCATE, { { "size", sizeString } });
				addOperation(SyntheticGPUOperation::DEVICE_ALLOCATE, { { "size", sizeString } });

				// Copy vectors to device
				addOperation(SyntheticGPUOperation::COPY_HOST_TO_DEVICE, { { "count", "2" } });

				// Do vector add
				addOperation(SyntheticGPUOperation::DEVICE_ADD, { { "count", "2" }, { "threadsPerBlock", sizeString } });

				// Copy results to host
				addOperation(SyntheticGPUOperation::COPY_DEVICE_TO_HOST, { { "count", "1" } });

				// Free device vectors
				addOperation(SyntheticGPUOperation::DEVICE_FREE, {});
				addOperation(SyntheticGPUOperation::DEVICE_FREE, {});
				addOperation(SyntheticGPUOperation::DEVICE_FREE, {});

				// Free host vectors
				addOperation(SyntheticGPUOperation::HOST_FREE, {});
				addOperation(SyntheticGPUOperation::HOST_FREE, {});
				addOperation(SyntheticGPUOperation::HOST_FREE, {});
			}
		}
	}
}