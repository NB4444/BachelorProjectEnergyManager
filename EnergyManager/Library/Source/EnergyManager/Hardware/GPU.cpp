#include "./GPU.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Units/Byte.hpp"

#include <algorithm>
#include <unistd.h>

/**
 * More performance variables to monitor can be found at these sources:
 * | Tool  | Functionality        | URL                                                                   |
 * | :---- | :------------------- | :-------------------------------------------------------------------- |
 * | CUPTI | Information          | https://docs.nvidia.com/cuda/cupti/index.html                         |
 * | CUPTI | Documentation        | https://docs.nvidia.com/cupti/Cupti/index.html                        |
 * | NVML  | Information          | https://developer.nvidia.com/nvidia-management-library-nvml           |
 * | NVML  | Device Queries       | https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html |
 * | CUDA  | Occupancy Calculator | https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html     |
 * | PAPI  | Information          | https://developer.nvidia.com/papi-cuda-component                      |
 */

namespace EnergyManager {
	namespace Hardware {
		Utility::StaticInitializer GPU::initializer_ = Utility::StaticInitializer([] {
			// Initialize CUDA
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuInit(0));

			// Get the device count to create a device context, which is necessary
			int deviceCount = 0;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaGetDeviceCount(&deviceCount));

			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaSetDeviceFlags(cudaDeviceBlockingSync));

			// Enable collection of various types of parameters
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE)); // DEVICE needs to be enabled before all others
			try {
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
			} catch(const std::exception& exception) {
			}
			try {
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT));
			} catch(const std::exception& exception) {
			}
			try {
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_KERNEL));
			} catch(const std::exception& exception) {
			}

			// Register callbacks
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityRegisterCallbacks(allocateBuffer, freeBuffer));

			// Initialize NVML
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlInit());
		});

		std::mutex GPU::monitorThreadMutex_;

		uint8_t* GPU::alignBuffer(uint8_t* buffer, const size_t& alignSize) {
			return (((uintptr_t)(buffer) & ((alignSize) -1)) ? ((buffer) + (alignSize) - ((uintptr_t)(buffer) & ((alignSize) -1))) : (buffer));
		}

		void CUPTIAPI GPU::allocateBuffer(uint8_t** buffer, size_t* size, size_t* maximumRecordCount) {
			// The size of the buffer used to collect statistics
			const size_t bufferSize = 32 * 1024;

			// The buffer's alignment
			const size_t alignSize = 8;

			auto* unalignedBuffer = (uint8_t*) malloc(bufferSize + alignSize);
			if(unalignedBuffer == nullptr) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Out of memory");
			}

			*size = bufferSize;
			*buffer = alignBuffer(unalignedBuffer, alignSize);
			*maximumRecordCount = 0;
		}

		void CUPTIAPI GPU::freeBuffer(CUcontext context, unsigned int streamId, uint8_t* buffer, size_t size, size_t validSize) {
			CUptiResult status;
			CUpti_Activity* record = nullptr;

			if(validSize > 0) {
				do {
					status = cuptiActivityGetNextRecord(buffer, validSize, &record);
					if(status == CUPTI_SUCCESS) {
						forwardActivity(record);
					} else if(status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
						break;
					} else {
						ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(status);
					}
				} while(true);

				// Report any records dropped from the queue
				size_t dropped;
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityGetNumDroppedRecords(context, streamId, &dropped));
				if(dropped != 0) {
					Utility::Logging::logInformation("Dropped %u activity records", static_cast<unsigned int>(dropped));
				}
			}

			free(buffer);
		}

		void GPU::forwardActivity(const CUpti_Activity* activity) {
			switch(activity->kind) {
				case CUPTI_ACTIVITY_KIND_DEVICE: {
					auto deviceActivity = (CUpti_ActivityDevice2*) activity;
					getGPU(deviceActivity->id)->handleDeviceActivity(deviceActivity);
					break;
				}
				case CUPTI_ACTIVITY_KIND_KERNEL:
				case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
					auto kernelActivity = (CUpti_ActivityKernel4*) activity;
					getGPU(kernelActivity->deviceId)->handleKernelActivity(kernelActivity);
					break;
				}
				default: {
					break;
				}
			}
		}

		void GPU::handleDeviceActivity(const CUpti_ActivityDevice2* activity) {
			memoryBandwidth_ = { Utility::Units::Byte(activity->globalMemoryBandwidth, Utility::Units::SIPrefix::KILO), std::chrono::seconds(1) };
			multiprocessorCount_ = activity->numMultiprocessors;
		}

		void GPU::handleKernelActivity(const CUpti_ActivityKernel4* activity) {
			kernelBlockX_ = activity->blockX;
			kernelBlockY_ = activity->blockY;
			kernelBlockZ_ = activity->blockZ;
			kernelContextID_ = activity->contextId;
			kernelCorrelationID_ = activity->correlationId;
			kernelDynamicSharedMemory_ = activity->dynamicSharedMemory;
			kernelEndTimestamp_ = activity->end;
			kernelGridX_ = activity->gridX;
			kernelGridY_ = activity->gridY;
			kernelGridZ_ = activity->gridZ;
			kernelName_ = activity->name;
			kernelStartTimestamp_ = activity->start;
			kernelStaticSharedMemory_ = activity->staticSharedMemory;
			kernelStreamID_ = activity->streamId;
		}

		GPU::GPU(const unsigned int& id) : Processor(id), Utility::Loopable(std::chrono::milliseconds(100)) {
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetHandleByIndex(id, &device_));

			// start the monitor thread
			run(true);
		}

		std::vector<std::string> GPU::generateHeaders() const {
			auto headers = Loopable::generateHeaders();
			headers.push_back("GPU " + Utility::Text::toString(getID()));

			return headers;
		}

		void GPU::onLoop() {
			const auto currentTimestamp = std::chrono::system_clock::now();

			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			energyConsumption_ += (static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastEnergyConsumptionPollTimestamp_).count()) / 1e3)
								  * getPowerConsumption().toValue();

			lastEnergyConsumptionPollTimestamp_ = currentTimestamp;
		}

		void GPU::handleAPICall(const std::string& call, const CUresult& callResult, const std::string& file, const int& line) {
			if(callResult != CUDA_SUCCESS) {
				throw Utility::Exceptions::Exception(
					"Driver call " + call + " failed with error code " + std::to_string(callResult) + ": " + cudaGetErrorString(static_cast<cudaError_t>(callResult)),
					file,
					line);
			}
		}

		void GPU::handleAPICall(const std::string& call, const cudaError_t& callResult, const std::string& file, const int& line) {
			if(callResult != static_cast<cudaError_t>(CUDA_SUCCESS)) {
				throw Utility::Exceptions::Exception("Runtime driver call " + call + " failed with error code " + std::to_string(callResult) + ": " + cudaGetErrorString(callResult), file, line);
			}
		}

		void GPU::handleAPICall(const std::string& call, const CUptiResult& callResult, const std::string& file, const int& line) {
			if(callResult != CUPTI_SUCCESS) {
				const char* errorMessage;
				cuptiGetResultString(callResult, &errorMessage);

				throw Utility::Exceptions::Exception("CUPTI call " + call + " failed with error code " + std::to_string(callResult) + ": " + errorMessage, file, line);
			}
		}

		void GPU::handleAPICall(const std::string& call, const nvmlReturn_t& callResult, const std::string& file, const int& line) {
			if(callResult != NVML_SUCCESS) {
				throw Utility::Exceptions::Exception("NVML call " + call + " failed with error code " + std::to_string(callResult) + ": " + nvmlErrorString(callResult), file, line);
			}
		}

		std::shared_ptr<GPU> GPU::getGPU(const unsigned int& id) {
			// Keep track of GPUs
			static std::map<unsigned int, std::shared_ptr<GPU>> gpus_ = {};

			auto iterator = gpus_.find(id);
			if(iterator == gpus_.end()) {
				gpus_[id] = std::shared_ptr<GPU>(new GPU(id));
			}

			return gpus_[id];
		}

		std::vector<std::shared_ptr<GPU>> GPU::getGPUs() {
			std::vector<std::shared_ptr<GPU>> gpus;
			for(unsigned int gpu = 0; gpu < getGPUCount(); ++gpu) {
				gpus.push_back(getGPU(gpu));
			}

			return gpus;
		}

		unsigned int GPU::getGPUCount() {
			int deviceCount = 0;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaGetDeviceCount(&deviceCount));

			return deviceCount;
		}

		void GPU::makeActive() const {
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaSetDevice(getID()));
		}

		Utility::Units::Hertz GPU::getApplicationCoreClockRate() const {
			unsigned int coreClockRate;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetApplicationsClock(device_, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, &coreClockRate));

			return { static_cast<double>(coreClockRate), Utility::Units::SIPrefix::MEGA };
		}

		void GPU::setApplicationCoreClockRate(const Utility::Units::Hertz& rate) {
			unsigned int memoryClockRate;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetApplicationsClock(device_, nvmlClockType_enum::NVML_CLOCK_MEM, &memoryClockRate));

			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceSetApplicationsClocks(device_, memoryClockRate, rate.convertPrefix(Utility::Units::SIPrefix::MEGA)));
		}

		void GPU::resetApplicationCoreClockRate() {
			auto memoryClockRate = getApplicationMemoryClockRate();

			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceResetApplicationsClocks(device_));

			setApplicationMemoryClockRate(memoryClockRate);
		}

		Utility::Units::Hertz GPU::getApplicationMemoryClockRate() const {
			unsigned int memoryClockRate;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetApplicationsClock(device_, nvmlClockType_enum::NVML_CLOCK_MEM, &memoryClockRate));

			return { static_cast<double>(memoryClockRate), Utility::Units::SIPrefix::MEGA };
		}

		void GPU::setApplicationMemoryClockRate(const Utility::Units::Hertz& rate) {
			unsigned int coreClockRate;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetApplicationsClock(device_, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, &coreClockRate));

			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceSetApplicationsClocks(device_, rate.convertPrefix(Utility::Units::SIPrefix::MEGA), coreClockRate));
		}

		void GPU::resetApplicationMemoryClockRate() {
			auto coreClockRate = getApplicationCoreClockRate();

			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceResetApplicationsClocks(device_));

			setApplicationCoreClockRate(coreClockRate);
		}

		bool GPU::getAutoBoostedClocksEnabled() const {
			nvmlEnableState_t autoBoostedClocksEnabled;
			nvmlEnableState_t defaultAutoBoostedClocksEnabled;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetAutoBoostedClocksEnabled(device_, &autoBoostedClocksEnabled, &defaultAutoBoostedClocksEnabled));

			return autoBoostedClocksEnabled == NVML_FEATURE_ENABLED;
		}

		bool GPU::getDefaultAutoBoostedClocksEnabled() const {
			nvmlEnableState_t autoBoostedClocksEnabled;
			nvmlEnableState_t defaultAutoBoostedClocksEnabled;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetAutoBoostedClocksEnabled(device_, &autoBoostedClocksEnabled, &defaultAutoBoostedClocksEnabled));

			return defaultAutoBoostedClocksEnabled == NVML_FEATURE_ENABLED;
		}

		void GPU::setAutoBoostedClocksEnabled(const bool& enabled) {
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(
				nvmlDeviceSetAutoBoostedClocksEnabled(device_, enabled ? nvmlEnableState_enum::NVML_FEATURE_ENABLED : nvmlEnableState_enum::NVML_FEATURE_DISABLED));
		}

		std::string GPU::getBrand() const {
			nvmlBrandType_t brand;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetBrand(device_, &brand));

			switch(brand) {
				case NVML_BRAND_GEFORCE:
					return "GeForce";
				case NVML_BRAND_GRID:
					return "Grid";
				case NVML_BRAND_NVS:
					return "NVS";
				case NVML_BRAND_QUADRO:
					return "Quadro";
				case NVML_BRAND_TESLA:
					return "Tesla";
				case NVML_BRAND_TITAN:
					return "Titan";
				case NVML_BRAND_UNKNOWN:
				default:
					return "Unknown";
			}
		}

		unsigned int GPU::getComputeCapabilityMajorVersion() const {
			int major;
			int minor;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetCudaComputeCapability(device_, &major, &minor));

			return major;
		}

		unsigned int GPU::getComputeCapabilityMinorVersion() const {
			int major;
			int minor;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetCudaComputeCapability(device_, &major, &minor));

			return minor;
		}

		Utility::Units::Hertz GPU::getCoreClockRate() const {
			unsigned int coreClockRate;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetClock(device_, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, nvmlClockId_enum::NVML_CLOCK_ID_CURRENT, &coreClockRate));

			return { static_cast<double>(coreClockRate), Utility::Units::SIPrefix::MEGA };
		}

		Utility::Units::Hertz GPU::getCurrentMinimumCoreClockRate() const {
			return currentMinimumCoreClockRate_;
		}

		Utility::Units::Hertz GPU::getCurrentMaximumCoreClockRate() const {
			return currentMaximumCoreClockRate_;
		}

		void GPU::setCoreClockRate(const Utility::Units::Hertz& mininimumRate, const Utility::Units::Hertz& maximumRate) {
			logDebug("Setting frequency range to [%lu, %lu]...", mininimumRate.toValue(), maximumRate.toValue());

			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(
				nvmlDeviceSetGpuLockedClocks(device_, mininimumRate.convertPrefix(Utility::Units::SIPrefix::MEGA), maximumRate.convertPrefix(Utility::Units::SIPrefix::MEGA)));

			currentMinimumCoreClockRate_ = mininimumRate;
			currentMaximumCoreClockRate_ = maximumRate;
		}

		std::vector<Utility::Units::Hertz> GPU::getSupportedCoreClockRates(const Utility::Units::Hertz& memoryClockRate) const {
			unsigned int count;
			unsigned int coreClockRates[100] { 0 };
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetSupportedGraphicsClocks(device_, memoryClockRate.convertPrefix(Utility::Units::SIPrefix::MEGA), &count, coreClockRates));

			std::vector<Utility::Units::Hertz> results = {};
			for(unsigned int index = 0; index < count; ++index) {
				results.emplace_back(coreClockRates[index]);
			}
			return results;
		}

		std::vector<Utility::Units::Hertz> GPU::getSupportedCoreClockRates() const {
			return getSupportedCoreClockRates(getMemoryClockRate());
		}

		void GPU::resetCoreClockRate() {
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceResetGpuLockedClocks(device_));
		}

		Utility::Units::Hertz GPU::getMinimumCoreClockRate() const {
			Utility::Units::Hertz minimumCoreClockRate;
			bool found = false;

			for(const auto& memoryClockRate : getSupportedMemoryClockRates()) {
				for(const auto& coreClockRate : getSupportedCoreClockRates(memoryClockRate)) {
					if(!found || coreClockRate < minimumCoreClockRate) {
						minimumCoreClockRate = coreClockRate;
						found = true;
					}
				}
			}

			return minimumCoreClockRate;
		}

		Utility::Units::Hertz GPU::getMaximumCoreClockRate() const {
			unsigned int maximumCoreClockRate;

			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetMaxClockInfo(device_, nvmlClockType_enum::NVML_CLOCK_GRAPHICS, &maximumCoreClockRate));

			return { static_cast<double>(maximumCoreClockRate), Utility::Units::SIPrefix::MEGA };
		}

		Utility::Units::Percent GPU::getCoreUtilizationRate() const {
			// Retrieve the utilization rates
			nvmlUtilization_t rates;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetUtilizationRates(device_, &rates));

			return rates.gpu;
		}

		Utility::Units::Joule GPU::getEnergyConsumption() const {
			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			return energyConsumption_;
		}

		Utility::Units::Watt GPU::getPowerConsumption() const {
			unsigned int powerConsumption = 0;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetPowerUsage(device_, &powerConsumption));

			return { static_cast<double>(powerConsumption), Utility::Units::SIPrefix::MILLI };
		}

		Utility::Units::Watt GPU::getPowerLimit() const {
			unsigned int powerManagementLimit;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetPowerManagementLimit(device_, &powerManagementLimit));

			return { static_cast<double>(powerManagementLimit), Utility::Units::SIPrefix::MILLI };
		}

		Utility::Units::Watt GPU::getDefaultPowerLimit() const {
			unsigned int defaultPowerManagementLimit;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetPowerManagementDefaultLimit(device_, &defaultPowerManagementLimit));

			return { static_cast<double>(defaultPowerManagementLimit), Utility::Units::SIPrefix::MILLI };
		}

		Utility::Units::Watt GPU::getEnforcedPowerLimit() const {
			unsigned int enforcedPowerLimit;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetEnforcedPowerLimit(device_, &enforcedPowerLimit));

			return { static_cast<double>(enforcedPowerLimit), Utility::Units::SIPrefix::MILLI };
		}

		Utility::Units::Percent GPU::getFanSpeed() const {
			unsigned int speed;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetFanSpeed(device_, &speed));

			return { static_cast<double>(speed) };
		}

		Utility::Units::Percent GPU::getFanSpeed(const unsigned int& fan) const {
			unsigned int speed;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetFanSpeed_v2(device_, fan, &speed));

			return { static_cast<double>(speed) };
		}

		int GPU::getKernelBlockX() const {
			return kernelBlockX_;
		}

		int GPU::getKernelBlockY() const {
			return kernelBlockY_;
		}

		int GPU::getKernelBlockZ() const {
			return kernelBlockZ_;
		}

		unsigned int GPU::getKernelContextID() const {
			return kernelContextID_;
		}

		unsigned int GPU::getKernelCorrelationID() const {
			return kernelCorrelationID_;
		}

		Utility::Units::Byte GPU::getKernelDynamicSharedMemorySize() const {
			return kernelDynamicSharedMemory_;
		}

		unsigned long GPU::getKernelEndTimestamp() const {
			return kernelEndTimestamp_;
		}

		int GPU::getKernelGridX() const {
			return kernelGridX_;
		}

		int GPU::getKernelGridY() const {
			return kernelGridY_;
		}

		int GPU::getKernelGridZ() const {
			return kernelGridZ_;
		}

		std::string GPU::getKernelName() const {
			return kernelName_;
		}

		unsigned long GPU::getKernelStartTimestamp() const {
			return kernelStartTimestamp_;
		}

		Utility::Units::Byte GPU::getKernelStaticSharedMemorySize() const {
			return kernelStaticSharedMemory_;
		}

		unsigned int GPU::getKernelStreamID() const {
			return kernelStreamID_;
		}

		Utility::Units::Hertz GPU::getMaximumMemoryClockRate() const {
			unsigned int maximumMemoryClockRate;

			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetMaxClockInfo(device_, nvmlClockType_enum::NVML_CLOCK_MEM, &maximumMemoryClockRate));

			return { static_cast<double>(maximumMemoryClockRate), Utility::Units::SIPrefix::MEGA };
		}

		Utility::Units::Byte GPU::getMemorySize() const {
			nvmlMemory_t memoryInfo;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetMemoryInfo(device_, &memoryInfo));

			return memoryInfo.total;
		}

		Utility::Units::Byte GPU::getMemoryFreeSize() const {
			nvmlMemory_t memoryInfo;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetMemoryInfo(device_, &memoryInfo));

			return memoryInfo.free;
		}

		Utility::Units::Byte GPU::getMemoryUsedSize() const {
			nvmlMemory_t memoryInfo;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetMemoryInfo(device_, &memoryInfo));

			return memoryInfo.used;
		}

		Utility::Units::Bandwidth GPU::getMemoryBandwidth() const {
			return memoryBandwidth_;
		}

		Utility::Units::Hertz GPU::getMemoryClockRate() const {
			unsigned int memoryClockRate;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetClock(device_, nvmlClockType_enum::NVML_CLOCK_MEM, nvmlClockId_enum::NVML_CLOCK_ID_CURRENT, &memoryClockRate));

			return { static_cast<double>(memoryClockRate), Utility::Units::SIPrefix::MEGA };
		}

		GPU::SynchronizationMode GPU::getSynchronizationMode() const {
			unsigned int flags;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaGetDeviceFlags(&flags));

			if(flags & cudaDeviceScheduleAuto) {
				return SynchronizationMode::AUTOMATIC;
			} else if(flags & cudaDeviceScheduleSpin) {
				return SynchronizationMode::SPIN;
			} else if(flags & cudaDeviceScheduleYield) {
				return SynchronizationMode::YIELD;
			} else if(flags & cudaDeviceScheduleBlockingSync) {
				return SynchronizationMode::BLOCKING;
			}

			// Return the default if none was explicitly set
			return SynchronizationMode::AUTOMATIC;
		}

		void GPU::setSynchronizationMode(const GPU::SynchronizationMode& synchronizationMode) {
			// Get the current flags
			unsigned int flags;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaGetDeviceFlags(&flags));

			// Unset any previous modes
			flags &= ~(cudaDeviceScheduleAuto | cudaDeviceScheduleSpin | cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync);

			// Set the mode
			switch(synchronizationMode) {
				case SynchronizationMode::AUTOMATIC:
					flags |= cudaDeviceScheduleAuto;
					break;
				case SynchronizationMode::SPIN:
					flags |= cudaDeviceScheduleSpin;
					break;
				case SynchronizationMode::YIELD:
					flags |= cudaDeviceScheduleYield;
					break;
				case SynchronizationMode::BLOCKING:
					flags |= cudaDeviceScheduleBlockingSync;
					break;
			}

			// Set the flags
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaSetDeviceFlags(flags));
		}

		std::vector<Utility::Units::Hertz> GPU::getSupportedMemoryClockRates() const {
			unsigned int count;
			unsigned int memoryClockRates[100] { 0 };
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetSupportedMemoryClocks(device_, &count, memoryClockRates));

			std::vector<Utility::Units::Hertz> results = {};
			for(unsigned int index = 0; index < count; ++index) {
				results.push_back(memoryClockRates[index]);
			}
			return results;
		}

		Utility::Units::Percent GPU::getMemoryUtilizationRate() const {
			// Retrieve the utilization rates
			nvmlUtilization_t rates;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetUtilizationRates(device_, &rates));

			return rates.memory;
		}

		unsigned int GPU::getMultiprocessorCount() const {
			return multiprocessorCount_;
		}

		std::string GPU::getName() const {
			char name[100] { '\0' };
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetName(device_, name, sizeof(name)));

			return name;
		}

		Utility::Units::Byte GPU::getPCIELinkWidth() const {
			unsigned int linkWidth;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetCurrPcieLinkWidth(device_, &linkWidth));

			return linkWidth;
		}

		Utility::Units::Hertz GPU::getStreamingMultiprocessorClockRate() const {
			unsigned int streamingMultiprocessorClockRate;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetClock(device_, nvmlClockType_enum::NVML_CLOCK_SM, nvmlClockId_enum::NVML_CLOCK_ID_CURRENT, &streamingMultiprocessorClockRate));

			return { static_cast<double>(streamingMultiprocessorClockRate), Utility::Units::SIPrefix::MEGA };
		}

		Utility::Units::Celsius GPU::getTemperature() const {
			unsigned int temperature;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetTemperature(device_, nvmlTemperatureSensors_enum::NVML_TEMPERATURE_GPU, &temperature));

			return temperature;
		}

		void GPU::reset() {
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaDeviceReset());
		}
	}
}