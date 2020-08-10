#include "./GPU.hpp"

#include "EnergyManager/Utility/Logging.hpp"

#include <algorithm>

namespace EnergyManager {
	namespace Hardware {
		const size_t GPU::bufferSize_;

		const size_t GPU::alignSize_;

		std::map<uint32_t, std::shared_ptr<GPU>> GPU::gpus_;

		uint8_t* GPU::alignBuffer(uint8_t* buffer, const size_t& alignSize) {
			return (((uintptr_t)(buffer) & ((alignSize) -1)) ? ((buffer) + (alignSize) - ((uintptr_t)(buffer) & ((alignSize) -1))) : (buffer));
		}

		void CUPTIAPI GPU::allocateBuffer(uint8_t** buffer, size_t* size, size_t* maximumRecordCount) {
			auto* unalignedBuffer = (uint8_t*) malloc(bufferSize_ + alignSize_);
			if(unalignedBuffer == nullptr) {
				throw std::runtime_error("Out of memory");
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
						forwardActivity(record);
					} else if(status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
						break;
					} else {
						HARDWARE_GPU_HANDLE_API_CALL(status);
					}
				} while(true);

				// Report any records dropped from the queue
				size_t dropped;
				HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityGetNumDroppedRecords(context, streamId, &dropped));
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
				case CUPTI_ACTIVITY_KIND_ENVIRONMENT: {
					auto environmentActivity = (CUpti_ActivityEnvironment*) activity;
					getGPU(environmentActivity->deviceId)->handleEnvironmentActivity(environmentActivity);
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
			computeCapabilityMajorVersion_ = activity->computeCapabilityMajor;
			computeCapabilityMinorVersion_ = activity->computeCapabilityMinor;
			coreClockRate_ = activity->coreClockRate;
			globalMemoryBandwidth_ = activity->globalMemoryBandwidth;
			globalMemorySize_ = activity->globalMemorySize;
			multiprocessorCount_ = activity->numMultiprocessors;
			name_ = activity->name;

			//switch(activity->kind) {
			//	case CUPTI_ACTIVITY_KIND_CONTEXT: {
			//		CUpti_ActivityContext* context = (CUpti_ActivityContext*) record;
			//		printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n", context->contextId, context->deviceId, getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind), (int) context->nullStreamId);
			//		break;
			//	}
			//	case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE: {
			//		CUpti_ActivityDeviceAttribute* attribute = (CUpti_ActivityDeviceAttribute*) record;
			//		printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n", attribute->attribute.cupti, attribute->deviceId, (unsigned long long) attribute->value.vUint64);
			//		break;
			//	}
			//	case CUPTI_ACTIVITY_KIND_DRIVER: {
			//		CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) record;
			//		printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n", api->cbid, (unsigned long long) (api->start - startTimestamp), (unsigned long long) (api->end - startTimestamp), api->processId, api->threadId, api->correlationId);
			//		break;
			//	}
			//	case CUPTI_ACTIVITY_KIND_MARKER: {
			//		CUpti_ActivityMarker2* marker = (CUpti_ActivityMarker2*) record;
			//		printf("MARKER id %u [ %llu ], name %s, domain %s\n", marker->id, (unsigned long long) marker->timestamp, marker->name, marker->domain);
			//		break;
			//	}
			//	case CUPTI_ACTIVITY_KIND_MARKER_DATA: {
			//		CUpti_ActivityMarkerData* marker = (CUpti_ActivityMarkerData*) record;
			//		printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n", marker->id, marker->color, marker->category, (unsigned long long) marker->payload.metricValueUint64, marker->payload.metricValueDouble);
			//		break;
			//	}
			//	case CUPTI_ACTIVITY_KIND_MEMCPY: {
			//		CUpti_ActivityMemcpy* memcpy = (CUpti_ActivityMemcpy*) record;
			//		printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n", getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind), (unsigned long long) (memcpy->start - startTimestamp), (unsigned long long) (memcpy->end - startTimestamp), memcpy->deviceId, memcpy->contextId, memcpy->streamId, memcpy->correlationId, memcpy->runtimeCorrelationId);
			//		break;
			//	}
			//	case CUPTI_ACTIVITY_KIND_MEMSET: {
			//		CUpti_ActivityMemset* memset = (CUpti_ActivityMemset*) record;
			//		printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n", memset->value, (unsigned long long) (memset->start - startTimestamp), (unsigned long long) (memset->end - startTimestamp), memset->deviceId, memset->contextId, memset->streamId, memset->correlationId);
			//		break;
			//	}
			//	case CUPTI_ACTIVITY_KIND_NAME: {
			//		CUpti_ActivityName* name = (CUpti_ActivityName*) record;
			//		switch(name->objectKind) {
			//			case CUPTI_ACTIVITY_OBJECT_CONTEXT:
			//				printf("NAME  %s %u %s id %u, name %s\n", getActivityObjectKindString(name->objectKind), getActivityObjectKindId(name->objectKind, &name->objectId), getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE), getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId), name->name);
			//				break;
			//			case CUPTI_ACTIVITY_OBJECT_STREAM:
			//				printf("NAME %s %u %s %u %s id %u, name %s\n", getActivityObjectKindString(name->objectKind), getActivityObjectKindId(name->objectKind, &name->objectId), getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT), getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId), getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE), getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId), name->name);
			//				break;
			//			default:
			//				printf("NAME %s id %u, name %s\n", getActivityObjectKindString(name->objectKind), getActivityObjectKindId(name->objectKind, &name->objectId), name->name);
			//				break;
			//		}
			//		break;
			//	}
			//	case CUPTI_ACTIVITY_KIND_OVERHEAD: {
			//		CUpti_ActivityOverhead* overhead = (CUpti_ActivityOverhead*) record;
			//		printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n", getActivityOverheadKindString(overhead->overheadKind), (unsigned long long) overhead->start - startTimestamp, (unsigned long long) overhead->end - startTimestamp, getActivityObjectKindString(overhead->objectKind), getActivityObjectKindId(overhead->objectKind, &overhead->objectId));
			//		break;
			//	}
			//	case CUPTI_ACTIVITY_KIND_RUNTIME: {
			//		CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) record;
			//		printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n", api->cbid, (unsigned long long) (api->start - startTimestamp), (unsigned long long) (api->end - startTimestamp), api->processId, api->threadId, api->correlationId);
			//		break;
			//	}
			//	default:
			//		break;
			//}
		}

		void GPU::handleEnvironmentActivity(const CUpti_ActivityEnvironment* activity) {
			switch(activity->environmentKind) {
				case CUPTI_ACTIVITY_ENVIRONMENT_COOLING:
					fanSpeed_ = activity->data.cooling.fanSpeed;
					break;
				case CUPTI_ACTIVITY_ENVIRONMENT_POWER:
					powerConsumption_ = activity->data.power.power;
					powerLimit_ = activity->data.power.powerLimit;
					break;
				case CUPTI_ACTIVITY_ENVIRONMENT_SPEED:
					memoryClock_ = activity->data.speed.memoryClock;
					streamingMultiprocessorClock_ = activity->data.speed.smClock;
					break;
				case CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE:
					temperature_ = activity->data.temperature.gpuTemperature;
					break;
				default:
					break;
			}
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

		GPU::GPU(const uint32_t& id)
			: id_(id) {
		}

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

		void GPU::initializeTracing() {
			// Initialize CUDA
			HARDWARE_GPU_HANDLE_API_CALL(cuInit(0));

			// Get the device count to create a device context, which is necessary
			int deviceCount = 0;
			HARDWARE_GPU_HANDLE_API_CALL(cudaGetDeviceCount(&deviceCount));

			// Enable collection of various types of parameters
			HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE)); // DEVICE needs to be enabled before all others
			HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
			//HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONTEXT));
			//HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER));
			HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT));
			HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_KERNEL));
			//HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MARKER));
			//HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY));
			//HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMSET));
			//HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_NAME));
			//HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OVERHEAD));
			//HANDLE_API_CALL(cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME));

			// Register callbacks
			HARDWARE_GPU_HANDLE_API_CALL(cuptiActivityRegisterCallbacks(allocateBuffer, freeBuffer));
		}

		std::shared_ptr<GPU> GPU::getGPU(const uint32_t& id) {
			auto iterator = gpus_.find(id);
			if(iterator == gpus_.end()) {
				gpus_[id] = std::shared_ptr<GPU>(new GPU(id));
			}

			return gpus_[id];
		}

		uint32_t GPU::getComputeCapabilityMajorVersion() const {
			return computeCapabilityMajorVersion_;
		}

		uint32_t GPU::getComputeCapabilityMinorVersion() const {
			return computeCapabilityMinorVersion_;
		}

		uint32_t GPU::getCoreClockRate() const {
			return coreClockRate_;
		}

		uint32_t GPU::getFanSpeed() const {
			return fanSpeed_;
		}

		uint64_t GPU::getGlobalMemoryBandwidth() const {
			return globalMemoryBandwidth_;
		}

		uint64_t GPU::getGlobalMemorySize() const {
			return globalMemorySize_;
		}

		uint32_t GPU::getID() const {
			return id_;
		}

		int32_t GPU::getKernelBlockX() const {
			return kernelBlockX_;
		}

		int32_t GPU::getKernelBlockY() const {
			return kernelBlockY_;
		}

		int32_t GPU::getKernelBlockZ() const {
			return kernelBlockZ_;
		}

		uint32_t GPU::getKernelContextID() const {
			return kernelContextID_;
		}

		uint32_t GPU::getKernelCorrelationID() const {
			return kernelCorrelationID_;
		}

		int32_t GPU::getKernelDynamicSharedMemory() const {
			return kernelDynamicSharedMemory_;
		}

		uint64_t GPU::getKernelEndTimestamp() const {
			return kernelEndTimestamp_;
		}

		int32_t GPU::getKernelGridX() const {
			return kernelGridX_;
		}

		int32_t GPU::getKernelGridY() const {
			return kernelGridY_;
		}

		int32_t GPU::getKernelGridZ() const {
			return kernelGridZ_;
		}

		std::string GPU::getKernelName() const {
			return kernelName_;
		}

		uint64_t GPU::getKernelStartTimestamp() const {
			return kernelStartTimestamp_;
		}

		int32_t GPU::getKernelStaticSharedMemory() const {
			return kernelStaticSharedMemory_;
		}

		uint32_t GPU::getKernelStreamID() const {
			return kernelStreamID_;
		}

		uint32_t GPU::getMemoryClock() const {
			return memoryClock_;
		}

		uint32_t GPU::getMultiprocessorCount() const {
			return multiprocessorCount_;
		}

		std::string GPU::getName() const {
			return name_;
		}

		uint32_t GPU::getPowerConsumption() const {
			return powerConsumption_;
		}

		uint32_t GPU::getPowerLimit() const {
			return powerLimit_;
		}

		uint32_t GPU::getStreamingMultiprocessorClock() const {
			return streamingMultiprocessorClock_;
		}

		uint32_t GPU::getTemperature() const {
			return temperature_;
		}
	}
}