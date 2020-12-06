#include "./GPU.hpp"

#include "EnergyManager/Utility/Environment.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Units/Byte.hpp"

#include <algorithm>
#include <unistd.h>
#include <utility>

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

//#define CUPTI_CALL(call) \
//	do { \
//		CUptiResult _status = call; \
//		if(_status != CUPTI_SUCCESS) { \
//			const char* errstr; \
//			cuptiGetResultString(_status, &errstr); \
//			fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr); \
//			exit(-1); \
//		} \
//	} while(0)
//
//#define BUF_SIZE (32 * 1024)
//#define ALIGN_SIZE (8)
//#define ALIGN_BUFFER(buffer, align) (((uintptr_t)(buffer) & ((align) -1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align) -1))) : (buffer))

namespace EnergyManager {
	namespace Hardware {
		GPU::Kernel::Kernel(
			std::string name,
			const int& gridX,
			const int& gridY,
			const int& gridZ,
			const int& blockX,
			const int& blockY,
			const int& blockZ,
			const unsigned int& contextID,
			const unsigned int& correlationID,
			const unsigned int& streamID,
			const std::chrono::system_clock::time_point& startTimestamp,
			const std::chrono::system_clock::time_point& endTimestamp,
			Utility::Units::Byte dynamicSharedMemorySize,
			Utility::Units::Byte staticSharedMemorySize)
			: name_(std::move(name))
			, gridX_(gridX)
			, gridY_(gridY)
			, gridZ_(gridZ)
			, blockX_(blockX)
			, blockY_(blockY)
			, blockZ_(blockZ)
			, contextID_(contextID)
			, correlationID_(correlationID)
			, streamID_(streamID)
			, startTimestamp_(startTimestamp)
			, endTimestamp_(endTimestamp)
			, dynamicSharedMemorySize_(std::move(dynamicSharedMemorySize))
			, staticSharedMemorySize_(std::move(staticSharedMemorySize)) {
		}

		GPU::Kernel::Kernel(const CUpti_ActivityKernel4& kernelActivity)
			: Kernel(
				kernelActivity.name,
				kernelActivity.gridX,
				kernelActivity.gridY,
				kernelActivity.gridZ,
				kernelActivity.blockX,
				kernelActivity.blockY,
				kernelActivity.blockZ,
				kernelActivity.contextId,
				kernelActivity.correlationId,
				kernelActivity.streamId,
				std::chrono::system_clock::time_point(std::chrono::nanoseconds(kernelActivity.start)),
				std::chrono::system_clock::time_point(std::chrono::nanoseconds(kernelActivity.end)),
				Utility::Units::Byte(kernelActivity.dynamicSharedMemory),
				Utility::Units::Byte(kernelActivity.staticSharedMemory)) {
		}

		const std::string& GPU::Kernel::getName() const {
			return name_;
		}

		int GPU::Kernel::getGridX() const {
			return gridX_;
		}

		int GPU::Kernel::getGridY() const {
			return gridY_;
		}

		int GPU::Kernel::getGridZ() const {
			return gridZ_;
		}

		int GPU::Kernel::getBlockX() const {
			return blockX_;
		}

		int GPU::Kernel::getBlockY() const {
			return blockY_;
		}

		int GPU::Kernel::getBlockZ() const {
			return blockZ_;
		}

		unsigned int GPU::Kernel::getContextID() const {
			return contextID_;
		}

		unsigned int GPU::Kernel::getCorrelationID() const {
			return correlationID_;
		}

		unsigned int GPU::Kernel::getStreamID() const {
			return streamID_;
		}

		const std::chrono::system_clock::time_point& GPU::Kernel::getStartTimestamp() const {
			return startTimestamp_;
		}

		const std::chrono::system_clock::time_point& GPU::Kernel::getEndTimestamp() const {
			return endTimestamp_;
		}

		const Utility::Units::Byte& GPU::Kernel::getDynamicSharedMemorySize() const {
			return dynamicSharedMemorySize_;
		}

		const Utility::Units::Byte& GPU::Kernel::getStaticSharedMemorySize() const {
			return staticSharedMemorySize_;
		}

		Utility::StaticInitializer GPU::initializer_ = Utility::StaticInitializer(
			[] {
				initializeCUPTI();
				initializeCUDA();
				initializeNVML();
			},
			[] {
				//// Flush all buffers
				//Utility::Logging::logTrace("Flushing CUPTI buffers...");
				////ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaDeviceSynchronize());
				//ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE));
				//ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaDeviceReset());
			});

		std::vector<GPU::Kernel> GPU::kernels_ = {};

		void GPU::initializeCUDA() {
			Utility::Logging::logDebug("Initializing CUDA...");

			// Initialize CUDA
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuInit(0));

			// Get the device count to create a device context, which is necessary
			int deviceCount = 0;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaGetDeviceCount(&deviceCount));

			// Report device information
			for(unsigned int deviceID = 0; deviceID < deviceCount; ++deviceID) {
				CUdevice device;
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuDeviceGet(&device, deviceID));
				char deviceName[32];
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuDeviceGetName(deviceName, 32, device));

				Utility::Logging::logDebug("Found CUDA device with name %s", deviceName);
			}
		}

		void GPU::initializeNVML() {
			Utility::Logging::logDebug("Initializing NVML...");

			// Initialize NVML
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlInit());
		}

		void GPU::initializeCUPTI() {
			Utility::Logging::logDebug("Initializing CUPTI...");

			// Enable collection of various types of parameters
			for(const auto& activityKind : {
					CUPTI_ACTIVITY_KIND_DEVICE, // DEVICE needs to be enabled before all others
					CUPTI_ACTIVITY_KIND_CONTEXT,
					CUPTI_ACTIVITY_KIND_DRIVER,
					CUPTI_ACTIVITY_KIND_RUNTIME,
					CUPTI_ACTIVITY_KIND_MEMCPY,
					CUPTI_ACTIVITY_KIND_MEMSET,
					CUPTI_ACTIVITY_KIND_NAME,
					CUPTI_ACTIVITY_KIND_MARKER,
					CUPTI_ACTIVITY_KIND_KERNEL,
					//CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
					CUPTI_ACTIVITY_KIND_OVERHEAD
					//CUPTI_ACTIVITY_KIND_ENVIRONMENT
				}) {
				try {
					Utility::Logging::logTrace("Enabling activity tracing for activity kind %d...", activityKind);
					ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityEnable(activityKind));
				} catch(const Utility::Exceptions::Exception& exception) {
					exception.log();
					Utility::Logging::logWarning("Could not enable activity kind %d", activityKind);
				}
			}

			// Register callbacks
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityRegisterCallbacks(allocateBuffer, freeBuffer));

			size_t attributeValue = 0;
			size_t attributeValueSize = sizeof(size_t);
			// Get and set activity attributes.
			// Attributes can be set by the CUPTI client to change behavior of the activity API.
			// Some attributes require to be set before any CUDA context is created to be effective,
			// e.g. to be applied to all device buffer allocations (see documentation).
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attributeValueSize, &attributeValue));
			attributeValue *= 2;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attributeValueSize, &attributeValue));

			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attributeValueSize, &attributeValue));
			attributeValue *= 2;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attributeValueSize, &attributeValue));

			//size_t attrValue = 0, attrValueSize = sizeof(size_t);
			//// Device activity record is created when CUDA initializes, so we
			//// want to enable it before cuInit() or any CUDA runtime call.
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
			//// Enable all other activity record kinds.
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
			//CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
			//
			//// Register callbacks for buffer requests and for buffers completed by CUPTI.
			//CUPTI_CALL(cuptiActivityRegisterCallbacks(allocateBuffer, freeBuffer));
			//
			//// Get and set activity attributes.
			//// Attributes can be set by the CUPTI client to change behavior of the activity API.
			//// Some attributes require to be set before any CUDA context is created to be effective,
			//// e.g. to be applied to all device buffer allocations (see documentation).
			//CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
			//printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned) attrValue);
			//attrValue *= 2;
			//CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
			//
			//CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
			//printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned) attrValue);
			//attrValue *= 2;
			//CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
			//
			////CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
		}

		uint8_t* GPU::alignBuffer(uint8_t* buffer, const size_t& alignSize) {
			return (((uintptr_t)(buffer) & ((alignSize) -1)) ? ((buffer) + (alignSize) - ((uintptr_t)(buffer) & ((alignSize) -1))) : (buffer));
		}

		void CUPTIAPI GPU::allocateBuffer(uint8_t** buffer, size_t* size, size_t* maximumRecordCount) {
			// The size of the buffer used to collect statistics
			*size = Utility::Units::Byte(5 * 1024, Utility::Units::SIPrefix::KILO).toValue();

			Utility::Logging::logTrace("Allocating CUPTI buffer of %d bytes...", *size);

			// The buffer's alignment
			const size_t alignSize = 8;

			auto* unalignedBuffer = (uint8_t*) malloc(*size + alignSize);
			if(unalignedBuffer == nullptr) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Out of memory");
			}

			*buffer = alignBuffer(unalignedBuffer, alignSize);
			*maximumRecordCount = 0;

			//uint8_t* bfr = (uint8_t*) malloc(BUF_SIZE + ALIGN_SIZE);
			//if(bfr == NULL) {
			//	printf("Error: out of memory\n");
			//	exit(-1);
			//}
			//
			//*size = BUF_SIZE;
			//*buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
			//*maximumRecordCount = 0;
		}

		void CUPTIAPI GPU::freeBuffer(CUcontext context, unsigned int streamId, uint8_t* buffer, size_t size, size_t validSize) {
			Utility::Logging::logTrace("Freeing CUPTI buffer...");

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
					Utility::Logging::logWarning("Dropped %u activity records", static_cast<unsigned int>(dropped));
				}
			}

			free(buffer);

			//CUptiResult status;
			//CUpti_Activity* record = NULL;
			//
			//if(validSize > 0) {
			//	do {
			//		status = cuptiActivityGetNextRecord(buffer, validSize, &record);
			//		if(status == CUPTI_SUCCESS) {
			//			forwardActivity(record);
			//		} else if(status == CUPTI_ERROR_MAX_LIMIT_REACHED)
			//			break;
			//		else {
			//			CUPTI_CALL(status);
			//		}
			//	} while(1);
			//
			//	// report any records dropped from the queue
			//	size_t dropped;
			//	CUPTI_CALL(cuptiActivityGetNumDroppedRecords(context, streamId, &dropped));
			//	if(dropped != 0) {
			//		printf("Dropped %u activity records\n", (unsigned int) dropped);
			//	}
			//}
			//
			//free(buffer);
		}

		void GPU::forwardActivity(const CUpti_Activity* activity) {
			Utility::Logging::logTrace("Processing CUPTI activity with kind %d...", activity->kind);

			switch(activity->kind) {
				case CUPTI_ACTIVITY_KIND_DEVICE: {
					auto deviceActivity = (CUpti_ActivityDevice2*) activity;
					Utility::Logging::logTrace("Forwarding CUPTI device activity to device %d...", deviceActivity->id);
					getGPU(deviceActivity->id)->handleDeviceActivity(deviceActivity);
					break;
				}
				case CUPTI_ACTIVITY_KIND_KERNEL:
				case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
					auto kernelActivity = (CUpti_ActivityKernel4*) activity;
					Utility::Logging::logTrace("Forwarding CUPTI kernel activity to device %d...", kernelActivity->deviceId);
					getGPU(kernelActivity->deviceId)->handleKernelActivity(kernelActivity);
					break;
				}
				case CUPTI_ACTIVITY_KIND_DRIVER: {
					CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) activity;
					//printf("DRIVER cbid=%u process %u, thread %u, correlation %u\n", api->cbid, api->processId, api->threadId, api->correlationId);
					break;
				}
				case CUPTI_ACTIVITY_KIND_RUNTIME: {
					CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) activity;
					//printf("RUNTIME cbid=%u process %u, thread %u, correlation %u\n", api->cbid, api->processId, api->threadId, api->correlationId);
					break;
				}
				default: {
					Utility::Logging::logTrace("Ignored activity");
					break;
				}
			}
		}

		void GPU::handleDeviceActivity(const CUpti_ActivityDevice2* activity) {
			memoryBandwidth_ = { Utility::Units::Byte(activity->globalMemoryBandwidth, Utility::Units::SIPrefix::KILO), std::chrono::seconds(1) };
			multiprocessorCount_ = activity->numMultiprocessors;
		}

		void GPU::handleKernelActivity(const CUpti_ActivityKernel4* activity) {
			kernels_.push_back(Kernel(*activity));
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
			// Flush all buffers
			logTrace("Flushing CUPTI buffers...");
			//ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaDeviceSynchronize());
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE));
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaDeviceReset());
		}

		void GPU::handleAPICall(const std::string& call, const CUresult& callResult, const std::string& file, const int& line) {
			if(callResult != CUDA_SUCCESS) {
				Utility::Exceptions::Exception(
					"Driver call " + call + " failed with error code " + std::to_string(callResult) + ": " + cudaGetErrorString(static_cast<cudaError_t>(callResult)),
					file,
					line)
					.throwWithStacktrace();
			}
		}

		void GPU::handleAPICall(const std::string& call, const cudaError_t& callResult, const std::string& file, const int& line) {
			if(callResult != static_cast<cudaError_t>(CUDA_SUCCESS)) {
				Utility::Exceptions::Exception("Runtime driver call " + call + " failed with error code " + std::to_string(callResult) + ": " + cudaGetErrorString(callResult), file, line)
					.throwWithStacktrace();
			}
		}

		void GPU::handleAPICall(const std::string& call, const CUptiResult& callResult, const std::string& file, const int& line) {
			if(callResult != CUPTI_SUCCESS) {
				const char* errorMessage;
				cuptiGetResultString(callResult, &errorMessage);

				Utility::Exceptions::Exception("CUPTI call " + call + " failed with error code " + std::to_string(callResult) + ": " + errorMessage, file, line).throwWithStacktrace();
			}
		}

		void GPU::handleAPICall(const std::string& call, const nvmlReturn_t& callResult, const std::string& file, const int& line) {
			if(callResult != NVML_SUCCESS) {
				Utility::Exceptions::Exception("NVML call " + call + " failed with error code " + std::to_string(callResult) + ": " + nvmlErrorString(callResult), file, line).throwWithStacktrace();
			}
		}

		std::shared_ptr<GPU> GPU::getGPU(const unsigned int& id) {
			// Only allow one thread to get GPUs at a time
			static std::mutex mutex;
			std::lock_guard<std::mutex> guard(mutex);

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

		Utility::Units::Percent GPU::getCoreUtilizationRate() {
			// Retrieve the utilization rates
			nvmlUtilization_t rates;
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(nvmlDeviceGetUtilizationRates(device_, &rates));

			return rates.gpu;
		}

		Utility::Units::Joule GPU::getEnergyConsumption() {
			// Get the timestamp
			const auto currentTimestamp = std::chrono::system_clock::now();

			energyConsumption_
				+= (static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(currentTimestamp - lastEnergyConsumptionUpdate_).count()) / 1e3) * getPowerConsumption().toValue();
			lastEnergyConsumptionUpdate_ = currentTimestamp;

			return energyConsumption_;
		}

		Utility::Units::Watt GPU::getPowerConsumption() {
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

		std::vector<GPU::Kernel> GPU::getKernels() const {
			return kernels_;
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

		std::map<std::chrono::system_clock::time_point, std::vector<std::pair<std::string, GPU::EventSite>>> GPU::getEvents() const {
			std::map<std::chrono::system_clock::time_point, std::vector<std::pair<std::string, EventSite>>> events;

			logTrace("Looking for GPU events...");
			const auto csvFileName = "reporter.events.csv.tmp";
			if(Utility::Environment::fileExists(csvFileName)) {
				// Contact the EAR accounting tool and store the data in CSV format
				logTrace("Found event data, loading...");
				const auto reporterData = Utility::Text::readFile(csvFileName);

				// Make sure that the file ends with a newline to prevent reading incomplete data
				auto reporterDataLines = Utility::Text::splitToVector(reporterData, "\n");

				// Ensure that it is possible to do a complete read
				if(reporterDataLines.back().empty()) {
					const auto data = Utility::Text::parseTable(Utility::Text::join(reporterDataLines, "\n"), "\n", ";");

					// Process the events
					for(const auto& event : data) {
						const auto timestamp = Utility::Text::timestampFromString(event.at("Timestamp"));
						const auto eventName = event.at("Event");

						logTrace("Processing event that occurred at %s with name %s", Utility::Text::formatTimestamp(timestamp).c_str(), eventName.c_str());

						// Create the timestamp data if it does not exist
						if(events.find(timestamp) == events.end()) {
							events[timestamp] = {};
						}

						// Add the current data
						events[timestamp].push_back({ eventName, event.at("Site") == "ENTER" ? EventSite::ENTER : EventSite::EXIT });
					}
				} else {
					logWarning("Reporter events file is incomplete");
				}
			} else {
				logWarning("Reporter events file does not exist");
			}

			return events;
		}

		void GPU::reset() {
			ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaDeviceReset());
		}
	}
}