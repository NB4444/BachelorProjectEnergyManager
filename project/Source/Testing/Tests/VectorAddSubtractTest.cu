#include "./VectorAddSubtractTest.hpp"

#include "Hardware/GPU.hpp"
#include "Testing/TestResults.hpp"

namespace Testing {
	namespace Tests {
		__global__ void vectorAdd(const int* A, const int* B, int* C, int N) {
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if(i < N)
				C[i] = A[i] + B[i];
		}

		__global__ void vectorSubtract(const int* A, const int* B, int* C, int N) {
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if(i < N)
				C[i] = A[i] - B[i];
		}

		void VectorAddSubtractTest::doPass(cudaStream_t stream) const {
			// Configuration
			size_t size = computeCount_ * sizeof(int);
			int threadsPerBlock = 256;
			int blocksPerGrid = (computeCount_ + threadsPerBlock - 1) / threadsPerBlock;

			// Allocate input vectors h_A and h_B in host memory
			// Don't bother to initialize
			int* hostVectorA = (int*) malloc(size);
			int* hostVectorB = (int*) malloc(size);
			int* hostVectorC = (int*) malloc(size);
			int* deviceVectorA;
			int* deviceVectorB;
			int* deviceVectorC;

			// Allocate vectors in device memory
			HARDWARE_GPU_HANDLE_API_CALL(cudaMalloc((void**) &deviceVectorA, size));
			HARDWARE_GPU_HANDLE_API_CALL(cudaMalloc((void**) &deviceVectorB, size));
			HARDWARE_GPU_HANDLE_API_CALL(cudaMalloc((void**) &deviceVectorC, size));

			HARDWARE_GPU_HANDLE_API_CALL(cudaMemcpyAsync(deviceVectorA, hostVectorA, size, cudaMemcpyHostToDevice, stream));
			HARDWARE_GPU_HANDLE_API_CALL(cudaMemcpyAsync(deviceVectorB, hostVectorB, size, cudaMemcpyHostToDevice, stream));

			// Run the kernels
			vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(deviceVectorA, deviceVectorB, deviceVectorC, computeCount_);
			vectorSubtract<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(deviceVectorA, deviceVectorB, deviceVectorC, computeCount_);

			HARDWARE_GPU_HANDLE_API_CALL(cudaMemcpyAsync(hostVectorC, deviceVectorC, size, cudaMemcpyDeviceToHost, stream));

			if(stream == 0) {
				HARDWARE_GPU_HANDLE_API_CALL(cudaDeviceSynchronize());
			} else {
				HARDWARE_GPU_HANDLE_API_CALL(cudaStreamSynchronize(stream));
			}

			free(hostVectorA);
			free(hostVectorB);
			free(hostVectorC);
			cudaFree(deviceVectorA);
			cudaFree(deviceVectorB);
			cudaFree(deviceVectorC);
		}

		TestResults VectorAddSubtractTest::onRun() {
			int devCount = 0;
			HARDWARE_GPU_HANDLE_API_CALL(cudaGetDeviceCount(&devCount));
			for(int deviceNum = 0; deviceNum < devCount; deviceNum++) {
				CUdevice device;
				HARDWARE_GPU_HANDLE_API_CALL(cuDeviceGet(&device, deviceNum));

				char deviceName[32];
				HARDWARE_GPU_HANDLE_API_CALL(cuDeviceGetName(deviceName, 32, device));
				printf("Device Name: %s\n", deviceName);

				HARDWARE_GPU_HANDLE_API_CALL(cudaSetDevice(deviceNum));
				// Do pass default stream
				doPass(0);

				// Do pass with user stream
				cudaStream_t stream0;
				HARDWARE_GPU_HANDLE_API_CALL(cudaStreamCreate(&stream0));
				doPass(stream0);

				cudaDeviceSynchronize();

				// Flush all remaining CUPTI buffers before resetting the device.
				// This can also be called in the cudaDeviceReset callback.
				cuptiActivityFlushAll(0);

				cudaDeviceReset();
			}

			return TestResults(*this, {});
		}

		VectorAddSubtractTest::VectorAddSubtractTest(const int& computeCount)
			: Test("VectorAddSubtractTest")
			, computeCount_(computeCount) {
		}
	}
}