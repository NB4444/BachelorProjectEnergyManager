#pragma once

#include "Hardware/Device.hpp"

#include <cuda.h>
#include <cupti.h>
#include <functional>
#include <map>

namespace Hardware {
	/**
	 * Represents a Graphics Processing Unit.
	 */
	class GPU : public Device {
		template<typename Type>
		using DeviceMap = std::map<uint32_t, Type>;

		/**
		 * The size of the buffer used to collect statistics.
		 */
		static const size_t bufferSize_ = 32 * 1024;

		/**
		 * The buffer's alignment.
		 */
		static const size_t alignSize_ = 8;

		/**
		 * The fan speed as percentage of maximum.
		 */
		static DeviceMap<uint32_t> fanSpeed_;

		/**
		 * The memory frequency in MHz.
		 */
		static DeviceMap<uint32_t> memoryClock_;

		/**
		 * The power in milliwatts consumed by GPU and associated circuitry.
		 */
		static DeviceMap<uint32_t> powerConsumption_;

		/**
		 * The power in milliwatts that will trigger power management algorithm.
		 */
		static DeviceMap<uint32_t> powerLimit_;

		/**
		 * The SM frequency in MHz.
		 */
		static DeviceMap<uint32_t> streamingMultiprocessorClock_;

		/**
		 * The GPU temperature in degrees C.
		 */
		static DeviceMap<uint32_t> temperature_;

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

		/**
		 * The ID of the device.
		 */
		uint32_t deviceID_;

	public:
		/**
		 * Initializes tracing capabilities.
		 */
		static void initializeTracing();

		/**
		 * Creates a new GPU.
		 * @param deviceID The ID of the device.
		 */
		GPU(const uint32_t& deviceID);

		/**
		 * The fan speed as percentage of maximum.
		 * @return The fan speed.
		 */
		uint32_t getFanSpeed() const;

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

		/**
		 * Gets the SM frequency in MHz.
		 * @return The streaming multiprocessor clock.
		 */
		uint32_t getStreamingMultiprocessorClock() const;

		/**
		 * Gets the GPU temperature in degrees C.
		 * @return The temperature.
		 */
		uint32_t getTemperature() const;
	};
}