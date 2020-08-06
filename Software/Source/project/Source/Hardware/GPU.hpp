#pragma once

#include "Hardware/Device.hpp"

#include <cuda.h>
#include <cupti.h>
#include <functional>

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
		 * The GPU temperature in degrees C.
		 */
		static uint32_t temperature_;

		/**
		 * The SM frequency in MHz.
		 */
		static uint32_t streamingMultiprocessorClock_;

		/**
		 * The memory frequency in MHz.
		 */
		static uint32_t memoryClock_;

		/**
		 * The power in milliwatts consumed by GPU and associated circuitry.
		 */
		static uint32_t powerConsumption_;

		/**
		 * The power in milliwatts that will trigger power management algorithm.
		 */
		static uint32_t powerLimit_;

		/**
		 * Handles the results of a call to the CUPTI library.
		 * @param callResult The result of the call.
		 */
		static void handleCUPTICall(const CUptiResult& callResult);

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
		 * Handles a new activity record.
		 * @param record The record.
		 */
		static void handleActivity(const CUpti_Activity* record);

	public:
		/**
		 * Initializes tracing capabilities.
		 */
		static void initializeTracing();

		/**
		 * Creates a new GPU.
		 */
		GPU() = default;

		/**
		 * Gets the GPU temperature in degrees C.
		 * @return The temperature.
		 */
		uint32_t getTemperature() const;

		/**
		 * Gets the SM frequency in MHz.
		 * @return The streaming multiprocessor clock.
		 */
		uint32_t getStreamingMultiprocessorClock() const;

		/**
		 * Gets the memory frequency in MHz.
		 * @return The memory clock.
		 */
		uint32_t getMemoryClock() const;

		/**
		 * Gets the power in milliwatts consumed by GPU and associated circuitry.
		 * @return The power consumption.
		 */
		uint32_t getPowerConsumption() const;

		/**
		 * Gets the power in milliwatts that will trigger power management algorithm.
		 * @return The power limit.
		 */
		uint32_t getPowerLimit() const;
	};
}