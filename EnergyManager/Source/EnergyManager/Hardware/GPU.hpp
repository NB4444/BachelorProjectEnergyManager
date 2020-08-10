#pragma once

#include "Device.hpp"

#include <cuda.h>
#include <cupti.h>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#define HARDWARE_GPU_HANDLE_API_CALL(CALL) \
	Hardware::GPU::handleAPICall(#CALL, CALL, __FILE__, __LINE__)

namespace EnergyManager {
	namespace Hardware {
		/**
		 * Represents a Graphics Processing Unit.
		 */
		class GPU : public Device {
			/**
			 * The size of the buffer used to collect statistics.
			 */
			static const size_t bufferSize_ = 32 * 1024;

			/**
			 * The buffer's alignment.
			 */
			static const size_t alignSize_ = 8;

			/**
			 * Keeps track of GPUs.
			 */
			static std::map<uint32_t, std::shared_ptr<GPU>> gpus_;

			/**
			 * Aligns the buffer with the configured parameters.
			 * @param buffer The buffer to align.
			 * @param alignSize The size by which to align.
			 * @return The aligned buffer.
			 */
			static uint8_t* alignBuffer(uint8_t* buffer, const size_t& alignSize);

			/**
			 * Allocates a new buffer for CUPTI data.
			 * @param buffer The buffer to allocate.
			 * @param size The size of the buffer.
			 * @param maximumRecordCount The maximum amount of records that can be stored in the buffer.
			 */
			static void CUPTIAPI allocateBuffer(uint8_t** buffer, size_t* size, size_t* maximumRecordCount);

			/**
			 * Frees an existing CUPTI data buffer.
			 * @param context The context to use.
			 * @param streamId The ID of the stream.
			 * @param buffer The buffer to free.
			 * @param size The size of the buffer.
			 * @param validSize ?
			 */
			static void CUPTIAPI freeBuffer(CUcontext context, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize);

			/**
			 * Forwards the activity to the corresponding GPU.
			 * @param activity The activity to forward.
			 */
			static void forwardActivity(const CUpti_Activity* activity);

			/**
			 * Compute capability for the device, major number.
			 */
			uint32_t computeCapabilityMajorVersion_;

			/**
			 * Compute capability for the device, minor number.
			 */
			uint32_t computeCapabilityMinorVersion_;

			/**
			 * The core clock rate of the device, in kHz.
			 */
			uint32_t coreClockRate_;

			/**
			 * The fan speed as percentage of maximum.
			 */
			uint32_t fanSpeed_;

			/**
			 * The global memory bandwidth available on the device, in kBytes/sec.
			 */
			uint64_t globalMemoryBandwidth_;

			/**
			 * The amount of global memory on the device, in bytes.
			 */
			uint64_t globalMemorySize_;

			/**
			 * The ID of the device.
			 */
			uint32_t id_;

			/**
			 * The X-dimension block size for the kernel.
			 */
			int32_t kernelBlockX_;

			/**
			 * The Y-dimension block size for the kernel.
			 */
			int32_t kernelBlockY_;

			/**
			 * The Z-dimension block size for the kernel.
			 */
			int32_t kernelBlockZ_;

			/**
			 * The ID of the context where the kernel is executing.
			 */
			uint32_t kernelContextID_;

			/**
			 * The correlation ID of the kernel.
			 * Each kernel execution is assigned a unique correlation ID that is identical to the correlation ID in the driver or runtime API activity record that launched the kernel.
			 */
			uint32_t kernelCorrelationID_;

			/**
			 * The dynamic shared memory reserved for the kernel, in bytes.
			 */
			int32_t kernelDynamicSharedMemory_;

			/**
			 * The end timestamp for the kernel execution, in ns.
			 * A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the kernel.
			 */
			uint64_t kernelEndTimestamp_;

			/**
			 * The X-dimension grid size for the kernel.
			 */
			int32_t kernelGridX_;

			/**
			 * The Y-dimension grid size for the kernel.
			 */
			int32_t kernelGridY_;

			/**
			 * The Z-dimension grid size for the kernel.
			 */
			int32_t kernelGridZ_;

			/**
			 * The name of the kernel.
			 * This name is shared across all activity records representing the same kernel, and so should not be modified.
			 */
			std::string kernelName_;

			/**
			 * The start timestamp for the kernel execution, in ns.
			 * A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the kernel.
			 */
			uint64_t kernelStartTimestamp_;

			/**
			 * The static shared memory allocated for the kernel, in bytes.
			 */
			int32_t kernelStaticSharedMemory_;

			/**
			 * The ID of the stream where the kernel is executing.
			 */
			uint32_t kernelStreamID_;

			/**
			 * The memory frequency in MHz.
			 */
			uint32_t memoryClock_;

			/**
			 * Number of multiprocessors on the device.
			 */
			uint32_t multiprocessorCount_;

			/**
			 * The device name.
			 * This name is shared across all activity records representing instances of the device, and so should not be modified.
			 */
			std::string name_;

			/**
			 * The power in milliwatts consumed by GPU and associated circuitry.
			 */
			uint32_t powerConsumption_;

			/**
			 * The power in milliwatts that will trigger power management algorithm.
			 */
			uint32_t powerLimit_;

			/**
			 * The SM frequency in MHz.
			 */
			uint32_t streamingMultiprocessorClock_;

			/**
			 * The GPU temperature in degrees C.
			 */
			uint32_t temperature_;

			/**
			 * Handles a device activity.
			 * @param activity The activity.
			 */
			void handleDeviceActivity(const CUpti_ActivityDevice2* activity);

			/**
			 * Handles an environment activity.
			 * @param activity The activity.
			 */
			void handleEnvironmentActivity(const CUpti_ActivityEnvironment* activity);

			/**
			 * Handles a kernel activity.
			 * @param activity The activity.
			 */
			void handleKernelActivity(const CUpti_ActivityKernel4* activity);

			/**
			 * Creates a new GPU.
			 * @param id The ID of the device.
			 */
			GPU(const uint32_t& id);

		public:
			/**
			 * Handles the results of a call to an API.
			 * @param call The call.
			 * @param callResult The result of the call.
			 * @param file The file.
			 * @param line The line.
			 */
			static void handleAPICall(const std::string& call, const CUresult& callResult, const std::string& file, const int& line);

			static void handleAPICall(const std::string& call, const cudaError_t& callResult, const std::string& file, const int& line);

			static void handleAPICall(const std::string& call, const CUptiResult& callResult, const std::string& file, const int& line);

			/**
			 * Initializes tracing capabilities.
			 */
			static void initializeTracing();

			/**
			 * Gets the GPU with the specified device ID.
			 * @param id The device ID.
			 * @return The GPU.
			 */
			static std::shared_ptr<GPU> getGPU(const uint32_t& id);

			/**
			 * @copydoc GPU::computeCapabilityMajorVersion_
			 * @return The compute capability major version.
			 */
			uint32_t getComputeCapabilityMajorVersion() const;

			/**
			 * @copydoc GPU::computeCapabilityMinorVersion_
			 * @return The compute capability minor version.
			 */
			uint32_t getComputeCapabilityMinorVersion() const;

			/**
			 * @copydoc GPU::coreClockRate_
			 * @return The core clock rate.
			 */
			uint32_t getCoreClockRate() const;

			/**
			 * @copydoc GPU::fanSpeed_
			 * @return The fan speed.
			 */
			uint32_t getFanSpeed() const;

			/**
			 * @copydoc GPU::globalMemoryBandwidth_
			 * @return The global memory bandwidth.
			 */
			uint64_t getGlobalMemoryBandwidth() const;

			/**
			 * @copydoc GPU::globalMemorySize_
			 * @return The global memory size.
			 */
			uint64_t getGlobalMemorySize() const;

			/**
			 * @copydoc GPU::id_
			 * @return The ID.
			 */
			uint32_t getID() const;

			/**
			 * @copydoc GPU::kernelBlockX_
			 * @return The kernel block X coordinate.
			 */
			int32_t getKernelBlockX() const;

			/**
			 * @copydoc GPU::kernelBlockY_
			 * @return The kernel block Y coordinate.
			 */
			int32_t getKernelBlockY() const;

			/**
			 * @copydoc GPU::kernelBlockZ_
			 * @return The kernel block Z coordinate.
			 */
			int32_t getKernelBlockZ() const;

			/**
			 * @copydoc GPU::kernelContextID_
			 * @return The kernel context ID.
			 */
			uint32_t getKernelContextID() const;

			/**
			 * @copydoc GPU::kernelCorrelationID_
			 * @return The kernel correlation ID.
			 */
			uint32_t getKernelCorrelationID() const;

			/**
			 * @copydoc GPU::kernelDynamicSharedMemory_
			 * @return The kernel dynamic shared memory.
			 */
			int32_t getKernelDynamicSharedMemory() const;

			/**
			 * @copydoc GPU::kernelEndTimestamp_
			 * @return The kernel end timestamp.
			 */
			uint64_t getKernelEndTimestamp() const;

			/**
			 * @copydoc GPU::kernelGridX_
			 * @return The kernel grid X coordinate.
			 */
			int32_t getKernelGridX() const;

			/**
			 * @copydoc GPU::kernelGridY_
			 * @return The kernel grid Y coordinate.
			 */
			int32_t getKernelGridY() const;

			/**
			 * @copydoc GPU::kernelGridZ_
			 * @return The kernel grid Z coordinate.
			 */
			int32_t getKernelGridZ() const;

			/**
			 * @copydoc GPU::kernelName_
			 * @return The kernel name.
			 */
			std::string getKernelName() const;

			/**
			 * @copydoc GPU::kernelStartTimestamp_
			 * @return The kernel start timestamp.
			 */
			uint64_t getKernelStartTimestamp() const;

			/**
			 * @copydoc GPU::kernelStaticSharedMemory_
			 * @return The kernel static shared memory.
			 */
			int32_t getKernelStaticSharedMemory() const;

			/**
			 * @copydoc GPU::kernelStreamID_
			 * @return The kernel stream ID.
			 */
			uint32_t getKernelStreamID() const;

			/**
			 * @copydoc GPU::memoryClock_
			 * @return The memory clock.
			 */
			uint32_t getMemoryClock() const;

			/**
			 * @copydoc GPU::multiprocessorCount_
			 * @return The multiprocessor count.
			 */
			uint32_t getMultiprocessorCount() const;

			/**
			 * @copydoc GPU::name_
			 * @return The name.
			 */
			std::string getName() const;

			/**
			 * @copydoc GPU::powerConsumption_
			 * @return The power consumption.
			 */
			uint32_t getPowerConsumption() const;

			/**
			 * @copydoc GPU::powerLimit_
			 * @return The power limit.
			 */
			uint32_t getPowerLimit() const;

			/**
			 * @copydoc GPU::streamingMultiprocessorClock_
			 * @return The streaming multiprocessor clock.
			 */
			uint32_t getStreamingMultiprocessorClock() const;

			/**
			 * @copydoc GPU::temperature_
			 * @return The temperature.
			 */
			uint32_t getTemperature() const;
		};
	}
}