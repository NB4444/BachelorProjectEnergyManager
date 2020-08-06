#include "./GPU.hpp"

#include "Utility/Logging.hpp"

#define HANDLE_API_CALL(CALL) \
	handleAPICall(#CALL, CALL, __FILE__, __LINE__)

namespace Hardware {
	const size_t GPU::bufferSize_;

	const size_t GPU::alignSize_;

	uint32_t GPU::temperature_;

	uint32_t GPU::streamingMultiprocessorClock_;

	uint32_t GPU::memoryClock_;

	uint32_t GPU::powerConsumption_;

	uint32_t GPU::powerLimit_;

	void GPU::handleAPICall(const std::string& call, const CUresult& callResult, const std::string& file, const int& line) {
		if(callResult != CUDA_SUCCESS) {
			Utility::Logging::logError("Driver call %s failed: %s", file, line, call.c_str(), callResult);
			//exit(-1);
		}
	}

	void GPU::handleAPICall(const std::string& call, const cudaError_t& callResult, const std::string& file, const int& line) {
		if(callResult != CUDA_SUCCESS) {
			Utility::Logging::logError("Runtime driver call %s failed: %s", file, line, call.c_str(), cudaGetErrorString(callResult));
			//exit(-1);
		}
	}

	void GPU::handleAPICall(const std::string& call, const CUptiResult& callResult, const std::string& file, const int& line) {
		if(callResult != CUPTI_SUCCESS) {
			const char* errorMessage;
			cuptiGetResultString(callResult, &errorMessage);
			Utility::Logging::logError("CUPTI call %s failed: %s", file, line, call.c_str(), errorMessage);
			//if(callResult == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
			//	exit(0);
			//} else {
			//	exit(-1);
			//}
		}
	}

	uint8_t* GPU::alignBuffer(uint8_t* buffer, const size_t& alignSize) {
		return (((uintptr_t)(buffer) & ((alignSize) -1)) ? ((buffer) + (alignSize) - ((uintptr_t)(buffer) & ((alignSize) -1))) : (buffer));
	}

	void CUPTIAPI GPU::allocateBuffer(uint8_t** buffer, size_t* size, size_t* maximumRecordCount) {
		auto* unalignedBuffer = (uint8_t*) malloc(bufferSize_ + alignSize_);
		if(unalignedBuffer == nullptr) {
			Utility::Logging::logError("Out of memory", __FILE__, __LINE__);
			exit(-1);
		}

		*size = bufferSize_;
		*buffer = alignBuffer(unalignedBuffer, alignSize_);
		*maximumRecordCount = 0;
	}

	void CUPTIAPI GPU::freeBuffer(CUcontext context, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize) {
		CUptiResult status;
		CUpti_Activity* record = nullptr;

		if(validSize > 0) {
			do {
				status = cuptiActivityGetNextRecord(buffer, validSize, &record);
				if(status == CUPTI_SUCCESS) {
					handleActivity(record);
				} else if(status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
					break;
				} else {
					HANDLE_API_CALL(status);
				}
			} while(true);

			// Report any records dropped from the queue
			size_t dropped;
			HANDLE_API_CALL(cuptiActivityGetNumDroppedRecords(context, streamId, &dropped));
			if(dropped != 0) {
				Utility::Logging::logInformation("Dropped %u activity records", static_cast<unsigned int>(dropped));
			}
		}

		free(buffer);
	}

	void GPU::handleActivity(const CUpti_Activity* record) {
		switch(record->kind) {
			//case CUPTI_ACTIVITY_KIND_CONTEXT: {
			//	CUpti_ActivityContext* context = (CUpti_ActivityContext*) record;
			//	printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n", context->contextId, context->deviceId, getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind), (int) context->nullStreamId);
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_DEVICE: {
			//	CUpti_ActivityDevice2* device = (CUpti_ActivityDevice2*) record;
			//	printf(
			//		"DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
			//		"multiprocessors %u, clock %u MHz\n",
			//		device->name,
			//		device->id,
			//		device->computeCapabilityMajor,
			//		device->computeCapabilityMinor,
			//		(unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
			//		(unsigned int) (device->globalMemorySize / 1024 / 1024),
			//		device->numMultiprocessors,
			//		(unsigned int) (device->coreClockRate / 1000));
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE: {
			//	CUpti_ActivityDeviceAttribute* attribute = (CUpti_ActivityDeviceAttribute*) record;
			//	printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n", attribute->attribute.cupti, attribute->deviceId, (unsigned long long) attribute->value.vUint64);
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_DRIVER: {
			//	CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) record;
			//	printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n", api->cbid, (unsigned long long) (api->start - startTimestamp), (unsigned long long) (api->end - startTimestamp), api->processId, api->threadId, api->correlationId);
			//	break;
			//}
			case CUPTI_ACTIVITY_KIND_ENVIRONMENT: {
				auto* environment = (CUpti_ActivityEnvironment*) record;

				switch(environment->environmentKind) {
					case CUPTI_ACTIVITY_ENVIRONMENT_POWER:
						powerConsumption_ = environment->data.power.power;
						powerLimit_ = environment->data.power.powerLimit;
						break;
					case CUPTI_ACTIVITY_ENVIRONMENT_SPEED:
						streamingMultiprocessorClock_ = environment->data.speed.smClock;
						memoryClock_ = environment->data.speed.memoryClock;
						break;
					case CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE:
						temperature_ = environment->data.temperature.gpuTemperature;
						break;
					default:
						break;
				}
			}
			//case CUPTI_ACTIVITY_KIND_KERNEL:
			//case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
			//	const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
			//	CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*) record;
			//	printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n", kindString, kernel->name, (unsigned long long) (kernel->start - startTimestamp), (unsigned long long) (kernel->end - startTimestamp), kernel->deviceId, kernel->contextId, kernel->streamId, kernel->correlationId);
			//	printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n", kernel->gridX, kernel->gridY, kernel->gridZ, kernel->blockX, kernel->blockY, kernel->blockZ, kernel->staticSharedMemory, kernel->dynamicSharedMemory);
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_MARKER: {
			//	CUpti_ActivityMarker2* marker = (CUpti_ActivityMarker2*) record;
			//	printf("MARKER id %u [ %llu ], name %s, domain %s\n", marker->id, (unsigned long long) marker->timestamp, marker->name, marker->domain);
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_MARKER_DATA: {
			//	CUpti_ActivityMarkerData* marker = (CUpti_ActivityMarkerData*) record;
			//	printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n", marker->id, marker->color, marker->category, (unsigned long long) marker->payload.metricValueUint64, marker->payload.metricValueDouble);
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_MEMCPY: {
			//	CUpti_ActivityMemcpy* memcpy = (CUpti_ActivityMemcpy*) record;
			//	printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n", getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind), (unsigned long long) (memcpy->start - startTimestamp), (unsigned long long) (memcpy->end - startTimestamp), memcpy->deviceId, memcpy->contextId, memcpy->streamId, memcpy->correlationId, memcpy->runtimeCorrelationId);
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_MEMSET: {
			//	CUpti_ActivityMemset* memset = (CUpti_ActivityMemset*) record;
			//	printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n", memset->value, (unsigned long long) (memset->start - startTimestamp), (unsigned long long) (memset->end - startTimestamp), memset->deviceId, memset->contextId, memset->streamId, memset->correlationId);
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_NAME: {
			//	CUpti_ActivityName* name = (CUpti_ActivityName*) record;
			//	switch(name->objectKind) {
			//		case CUPTI_ACTIVITY_OBJECT_CONTEXT:
			//			printf("NAME  %s %u %s id %u, name %s\n", getActivityObjectKindString(name->objectKind), getActivityObjectKindId(name->objectKind, &name->objectId), getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE), getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId), name->name);
			//			break;
			//		case CUPTI_ACTIVITY_OBJECT_STREAM:
			//			printf("NAME %s %u %s %u %s id %u, name %s\n", getActivityObjectKindString(name->objectKind), getActivityObjectKindId(name->objectKind, &name->objectId), getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT), getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId), getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE), getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId), name->name);
			//			break;
			//		default:
			//			printf("NAME %s id %u, name %s\n", getActivityObjectKindString(name->objectKind), getActivityObjectKindId(name->objectKind, &name->objectId), name->name);
			//			break;
			//	}
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_OVERHEAD: {
			//	CUpti_ActivityOverhead* overhead = (CUpti_ActivityOverhead*) record;
			//	printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n", getActivityOverheadKindString(overhead->overheadKind), (unsigned long long) overhead->start - startTimestamp, (unsigned long long) overhead->end - startTimestamp, getActivityObjectKindString(overhead->objectKind), getActivityObjectKindId(overhead->objectKind, &overhead->objectId));
			//	break;
			//}
			//case CUPTI_ACTIVITY_KIND_RUNTIME: {
			//	CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) record;
			//	printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n", api->cbid, (unsigned long long) (api->start - startTimestamp), (unsigned long long) (api->end - startTimestamp), api->processId, api->threadId, api->correlationId);
			//	break;
			//}
			default:
				break;
		}
	}

	void GPU::initializeTracing() {
		// Initialize CUDA
		HANDLE_API_CALL(cuInit(0));

		// Get the device count to create a device context, which is necessary
		int deviceCount = 0;
		HANDLE_API_CALL(cudaGetDeviceCount(&deviceCount));

		// Enable collection of various types of parameters
		HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE)); // DEVICE needs to be enabled before all others
		HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT));

		// Register callbacks
		HANDLE_API_CALL(cuptiActivityRegisterCallbacks(allocateBuffer, freeBuffer));

		// Print GPU information
		CUdevice device;
		char deviceName[32];
		for(int deviceNumber = 0; deviceNumber < deviceCount; deviceNumber++) {
			HANDLE_API_CALL(cuDeviceGet(&device, deviceNumber));
			HANDLE_API_CALL(cuDeviceGetName(deviceName, 32, device));
			printf("Device Name: %s\n", deviceName);

			HANDLE_API_CALL(cudaSetDevice(deviceNumber));
			// do pass default stream
			//do_pass(0);

			// do pass with user stream
			cudaStream_t stream0;
			HANDLE_API_CALL(cudaStreamCreate(&stream0));
			//do_pass(stream0);

			cudaDeviceSynchronize();

			// Flush all remaining CUPTI buffers before resetting the device.
			// This can also be called in the cudaDeviceReset callback.
			cuptiActivityFlushAll(0);

			cudaDeviceReset();
		}
	}

	uint32_t GPU::getTemperature() const {
		return temperature_;
	}

	uint32_t GPU::getStreamingMultiprocessorClock() const {
		return streamingMultiprocessorClock_;
	}

	uint32_t GPU::getMemoryClock() const {
		return memoryClock_;
	}

	uint32_t GPU::getPowerConsumption() const {
		return powerConsumption_;
	}

	uint32_t GPU::getPowerLimit() const {
		return powerLimit_;
	}
}