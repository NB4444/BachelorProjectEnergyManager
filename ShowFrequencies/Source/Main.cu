#include <cuda.h>
#include <fstream>
#include <iostream>
#include <nvml.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern "C" {
#include <ear.h>
}

unsigned int deviceID = 0;

__global__ void vectorAdd(const int* A, const int* B, int* C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < N) {
		C[i] = A[i] + B[i];
	}
}

__global__ void vectorSubtract(const int* A, const int* B, int* C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < N) {
		C[i] = A[i] - B[i];
	}
}

void doPass(cudaStream_t stream) {
	std::cout << "Doing pass..." << std::endl;

	const int computeCount = 50000;

	// Configuration
	size_t size = computeCount * sizeof(int);
	int threadsPerBlock = 256;
	int blocksPerGrid = (computeCount + threadsPerBlock - 1) / threadsPerBlock;

	// Allocate input vectors h_A and h_B in host memory
	// Don't bother to initialize
	int* hostVectorA = (int*) malloc(size);
	int* hostVectorB = (int*) malloc(size);
	int* hostVectorC = (int*) malloc(size);
	int* deviceVectorA;
	int* deviceVectorB;
	int* deviceVectorC;

	// Allocate vectors in device memory
	cudaMalloc((void**) &deviceVectorA, size);
	cudaMalloc((void**) &deviceVectorB, size);
	cudaMalloc((void**) &deviceVectorC, size);

	cudaMemcpyAsync(deviceVectorA, hostVectorA, size, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(deviceVectorB, hostVectorB, size, cudaMemcpyHostToDevice, stream);

	// Run the kernels
	vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(deviceVectorA, deviceVectorB, deviceVectorC, computeCount);
	vectorSubtract<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(deviceVectorA, deviceVectorB, deviceVectorC, computeCount);

	cudaMemcpyAsync(hostVectorC, deviceVectorC, size, cudaMemcpyDeviceToHost, stream);

	if(stream == 0) {
		cudaDeviceSynchronize();
	} else {
		cudaStreamSynchronize(stream);
	}

	free(hostVectorA);
	free(hostVectorB);
	free(hostVectorC);
	cudaFree(deviceVectorA);
	cudaFree(deviceVectorB);
	cudaFree(deviceVectorC);

	std::cout << "Did pass" << std::endl;
}

int main() {
	// Initialize CUDA
	cuInit(0);

	// Initialize NVML
	nvmlInit();

	// Get the device count to create a device context, which is necessary
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	// Get the current device
	nvmlDevice_t nvmlDevice;
	nvmlDeviceGetHandleByIndex(deviceID, &nvmlDevice);

	// Report device information
	CUdevice cudaDevice;
	cuDeviceGet(&cudaDevice, deviceID);
	char deviceName[32];
	cuDeviceGetName(deviceName, 32, cudaDevice);

	// Set the active device
	cudaSetDevice(deviceID);

	// Get supported clock speeds
	unsigned int memoryClockRate;
	nvmlDeviceGetClockInfo(nvmlDevice, nvmlClockType_enum::NVML_CLOCK_MEM, &memoryClockRate);
	unsigned int count;
	unsigned int coreClockRates[100] { 0 };
	nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice, memoryClockRate, &count, coreClockRates);
	for(unsigned int index = 0; index < count; ++index) {
		//std::cout << "Supported clock rate: " << coreClockRates[index] << std::endl;
	}

	//// Set the clock speeds
	//nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice, nvmlEnableState_enum::NVML_FEATURE_DISABLED);
	//nvmlDeviceSetApplicationsClocks(nvmlDevice, 1380000, 1380000);
	//nvmlDeviceSetGpuLockedClocks(nvmlDevice, 1380000, 1380000);

	// Start the monitor thread
	bool running = true;
	auto thread = std::thread([&] {
		while(running) {
			// CPU clock rates
			std::ifstream coreClockRateStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
			std::string coreClockRateString((std::istreambuf_iterator<char>(coreClockRateStream)), std::istreambuf_iterator<char>());
			auto cpuClockRate = std::stoul(coreClockRateString);

			// GPU clock rates
			unsigned int graphicsClockRate;
			nvmlDeviceGetClock(nvmlDevice, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, nvmlClockId_enum::NVML_CLOCK_ID_CURRENT, &graphicsClockRate);
			//unsigned int graphicsClockRate2;
			//nvmlDeviceGetClockInfo(nvmlDevice, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, &graphicsClockRate2);
			unsigned int graphicsApplicationClockRate;
			nvmlDeviceGetApplicationsClock(nvmlDevice, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, &graphicsApplicationClockRate);
			unsigned int smClockRate;
			nvmlDeviceGetClock(nvmlDevice, nvmlClockType_enum::NVML_CLOCK_SM, nvmlClockId_enum::NVML_CLOCK_ID_CURRENT, &smClockRate);
			//unsigned int smClockRate2;
			//nvmlDeviceGetClockInfo(nvmlDevice, nvmlClockType_enum::NVML_CLOCK_SM, &smClockRate2);
			unsigned int smApplicationClockRate;
			nvmlDeviceGetApplicationsClock(nvmlDevice, nvmlClockType_enum::NVML_CLOCK_SM, &smApplicationClockRate);

			std::cout << "CPU_CLOCK_RATE=" << cpuClockRate;
			std::cout << "    GRAPHICS_CLOCK_RATE=" << graphicsClockRate;
			//std::cout << "    GRAPHICS_CLOCK_RATE2=" << graphicsClockRate2;
			std::cout << "    GRAPHICS_APPLICATION_CLOCK_RATE=" << graphicsApplicationClockRate;
			std::cout << "    SM_CLOCK_RATE=" << smClockRate;
			//std::cout << "    SM_CLOCK_RATE2=" << smClockRate2;
			std::cout << "    SM_APPLICATION_CLOCK_RATE=" << smApplicationClockRate;
			std::cout << std::endl;

			usleep(1e6);
		}
	});

	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(0, &mask);

	ear_connect();

	std::cout << "Setting CPU frequency to 2.6 GHz..." << std::endl;
	std::cout << "Status: " << ear_set_cpufreq(&mask, 2600 * 1e3) << std::endl;

	for(unsigned int attempt = 0; attempt < 30; ++attempt) {
		// Do pass default stream
		doPass(0);

		// Do pass with user stream
		cudaStream_t stream0;
		cudaStreamCreate(&stream0);
		doPass(stream0);
	}
	cudaDeviceSynchronize();
	cudaDeviceReset();

	std::cout << "Waiting 10 seconds..." << std::endl;
	usleep(10 * 1e6);

	std::cout << "Setting CPU frequency to 2 GHz..." << std::endl;
	std::cout << "Status: " << ear_set_cpufreq(&mask, 2 * 1e6) << std::endl;

	for(unsigned int attempt = 0; attempt < 30; ++attempt) {
		// Do pass default stream
		doPass(0);

		// Do pass with user stream
		cudaStream_t stream0;
		cudaStreamCreate(&stream0);
		doPass(stream0);
	}
	cudaDeviceSynchronize();
	cudaDeviceReset();

	std::cout << "Waiting 10 seconds..." << std::endl;
	usleep(10 * 1e6);

	std::cout << "Setting GPU frequency to 135 MHz..." << std::endl;
	std::cout << "Status: " << ear_set_gpufreq(0, 135 * 1e3) << std::endl;

	for(unsigned int attempt = 0; attempt < 30; ++attempt) {
		// Do pass default stream
		doPass(0);

		// Do pass with user stream
		cudaStream_t stream0;
		cudaStreamCreate(&stream0);
		doPass(stream0);
	}
	cudaDeviceSynchronize();
	cudaDeviceReset();

	std::cout << "Waiting 10 seconds..." << std::endl;
	usleep(10 * 1e6);

	std::cout << "Setting GPU frequency to 1245 MHz..." << std::endl;
	std::cout << "Status: " << ear_set_gpufreq(0, 1245 * 1e3) << std::endl;

	for(unsigned int attempt = 0; attempt < 30; ++attempt) {
		// Do pass default stream
		doPass(0);

		// Do pass with user stream
		cudaStream_t stream0;
		cudaStreamCreate(&stream0);
		doPass(stream0);
	}
	cudaDeviceSynchronize();
	cudaDeviceReset();

	std::cout << "Waiting 10 seconds..." << std::endl;
	usleep(10 * 1e6);

	running = false;
	thread.join();

	ear_disconnect();

	//nvmlDevice_t device_;
	//nvmlDeviceGetHandleByIndex(0, &device_);
	//
	//unsigned int coreClockRate;
	//nvmlDeviceGetClock(device_, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, nvmlClockId_enum::NVML_CLOCK_ID_CURRENT, &coreClockRate);
	//
	//std::cout << coreClockRate;

	return 0;
}
