#include <cuda.h>
#include <iostream>
#include <nvml.h>

int main() {
	// Initialize CUDA
	cuInit(0);

	// Get the device count to create a device context, which is necessary
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	// Report device information
	for(unsigned int deviceID = 0; deviceID < deviceCount; ++deviceID) {
		CUdevice device;
		cuDeviceGet(&device, deviceID);
		char deviceName[32];
		cuDeviceGetName(deviceName, 32, device);
	}

	// Initialize NVML
	nvmlInit();

	for(unsigned int deviceID = 0; deviceID < deviceCount; ++deviceID) {
		nvmlDevice_t device_;
		nvmlDeviceGetHandleByIndex(deviceID, &device_);

		unsigned int graphicsClockRate;
		nvmlDeviceGetClock(device_, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, nvmlClockId_enum::NVML_CLOCK_ID_CURRENT, &graphicsClockRate);
		unsigned int smClockRate;
		nvmlDeviceGetClock(device_, nvmlClockType_enum::NVML_CLOCK_SM, nvmlClockId_enum::NVML_CLOCK_ID_CURRENT, &smClockRate);

		std::cout << "Device " << deviceID << " - GRAPHICS_CLOCK_RATE=" << graphicsClockRate << " SM_CLOCK_RATE=" << smClockRate << std::endl;
	}

	//nvmlDevice_t device_;
	//nvmlDeviceGetHandleByIndex(0, &device_);
	//
	//unsigned int coreClockRate;
	//nvmlDeviceGetClock(device_, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, nvmlClockId_enum::NVML_CLOCK_ID_CURRENT, &coreClockRate);
	//
	//std::cout << coreClockRate;

	return 0;
}
