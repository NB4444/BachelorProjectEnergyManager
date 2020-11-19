#pragma once

#include "EnergyManager/Hardware/Processor.hpp"
#include "EnergyManager/Utility/Loopable.hpp"
#include "EnergyManager/Utility/StaticInitializer.hpp"
#include "EnergyManager/Utility/Units/Bandwidth.hpp"
#include "EnergyManager/Utility/Units/Celsius.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"
#include "EnergyManager/Utility/Units/Percent.hpp"
#include "EnergyManager/Utility/Units/RotationsPerMinute.hpp"
#include "EnergyManager/Utility/Units/Watt.hpp"

#include <chrono>
#include <cuda.h>
#include <cupti.h>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <nvml.h>
#include <thread>
#include <vector>

#define ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(CALL) EnergyManager::Hardware::GPU::handleAPICall(#CALL, CALL, __FILE__, __LINE__)

namespace EnergyManager {
	namespace Hardware {
		/**
		 * Represents a Graphics Processing Unit.
		 */
		class GPU
			: public Processor
			, private Utility::Loopable {
			/**
			 * Initializes the APIs.
			 */
			static Utility::StaticInitializer initializer_;

			/**
			 * The mutex used to access variables that are recorded by the monitor thread.
			 */
			static std::mutex monitorThreadMutex_;

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
			static void CUPTIAPI freeBuffer(CUcontext context, unsigned int streamId, uint8_t* buffer, size_t size, size_t validSize);

			/**
			 * Forwards the activity to the corresponding GPU.
			 * @param activity The activity to forward.
			 */
			static void forwardActivity(const CUpti_Activity* activity);

			/**
			 * The GPU device.
			 */
			nvmlDevice_t device_;

			/**
			 * The energy consumption in Joules.
			 */
			Utility::Units::Joule energyConsumption_ = 0;

			/**
			 * Record the last time at which the energy consumption was polled.
			 */
			std::chrono::system_clock::time_point lastEnergyConsumptionPollTimestamp_ = std::chrono::system_clock::now();

			/**
			 * The global memory bandwidth available on the device, in kBytes/sec.
			 */
			Utility::Units::Bandwidth memoryBandwidth_;

			/**
			 * The X-dimension block size for the kernel.
			 */
			int kernelBlockX_;

			/**
			 * The Y-dimension block size for the kernel.
			 */
			int kernelBlockY_;

			/**
			 * The Z-dimension block size for the kernel.
			 */
			int kernelBlockZ_;

			/**
			 * The ID of the context where the kernel is executing.
			 */
			unsigned int kernelContextID_;

			/**
			 * The correlation ID of the kernel.
			 * Each kernel execution is assigned a unique correlation ID that is identical to the correlation ID in the driver or runtime API activity record that launched the kernel.
			 */
			unsigned int kernelCorrelationID_;

			/**
			 * The dynamic shared memory reserved for the kernel, in bytes.
			 */
			Utility::Units::Byte kernelDynamicSharedMemory_;

			/**
			 * The end timestamp for the kernel execution, in ns.
			 * A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the kernel.
			 */
			unsigned long kernelEndTimestamp_;

			/**
			 * The X-dimension grid size for the kernel.
			 */
			int kernelGridX_;

			/**
			 * The Y-dimension grid size for the kernel.
			 */
			int kernelGridY_;

			/**
			 * The Z-dimension grid size for the kernel.
			 */
			int kernelGridZ_;

			/**
			 * The name of the kernel.
			 * This name is shared across all activity records representing the same kernel, and so should not be modified.
			 */
			std::string kernelName_;

			/**
			 * The start timestamp for the kernel execution, in ns.
			 * A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the kernel.
			 */
			unsigned long kernelStartTimestamp_;

			/**
			 * The static shared memory allocated for the kernel, in bytes.
			 */
			Utility::Units::Byte kernelStaticSharedMemory_;

			/**
			 * The ID of the stream where the kernel is executing.
			 */
			unsigned int kernelStreamID_;

			/**
			 * Number of multiprocessors on the device.
			 */
			unsigned int multiprocessorCount_;

			/**
			 * The current minimum core clock rate.
			 */
			Utility::Units::Hertz currentMinimumCoreClockRate_;

			/**
			 * The current maximum core clock rate.
			 */
			Utility::Units::Hertz currentMaximumCoreClockRate_;

			/**
			 * Handles a device activity.
			 * @param activity The activity.
			 */
			void handleDeviceActivity(const CUpti_ActivityDevice2* activity);

			/**
			 * Handles a kernel activity.
			 * @param activity The activity.
			 */
			void handleKernelActivity(const CUpti_ActivityKernel4* activity);

			/**
			 * Creates a new GPU.
			 * @param id The ID of the device.
			 */
			explicit GPU(const unsigned int& id);

		protected:
			void onLoop() override;

		public:
			/**
			 * A mode of synchronization between the host and the device.
			 * Determines behavior when calling synchronization methods.
			 */
			enum class SynchronizationMode {
				/**
				 * The default value if the flags parameter is zero, uses a heuristic based on the number of active CUDA contexts in the process C and the number of logical processors in the system P.
				 * If C > P, then CUDA will yield to other OS threads when waiting for the device, otherwise CUDA will not yield while waiting for results and actively spin on the processor.
				 * Additionally, on Tegra devices, cudaDeviceScheduleAuto uses a heuristic based on the power profile of the platform and may choose cudaDeviceScheduleBlockingSync for low-powered devices.
				 */
				AUTOMATIC,

				/**
				 * Instruct CUDA to actively spin when waiting for results from the device.
				 * This can decrease latency when waiting for the device, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread.
				 */
				SPIN,

				/**
				 * Instruct CUDA to yield its thread when waiting for results from the device.
				 * This can increase latency when waiting for the device, but can increase the performance of CPU threads performing work in parallel with the device.
				 */
				YIELD,

				/**
				 * Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the device to finish work.
				 */
				BLOCKING
			};

			/**
			 * Handles the results of a call to the CUDA driver.
			 * @param call The call.
			 * @param callResult The result of the call.
			 * @param file The file.
			 * @param line The line.
			 */
			static void handleAPICall(const std::string& call, const CUresult& callResult, const std::string& file, const int& line);

			/**
			 * Handles the results of a call to an CUDA runtime driver.
			 * @param call The call.
			 * @param callResult The result of the call.
			 * @param file The file.
			 * @param line The line.
			 */
			static void handleAPICall(const std::string& call, const cudaError& callResult, const std::string& file, const int& line);

			/**
			 * Handles the results of a call to the CUPTI API.
			 * @param call The call.
			 * @param callResult The result of the call.
			 * @param file The file.
			 * @param line The line.
			 */
			static void handleAPICall(const std::string& call, const CUptiResult& callResult, const std::string& file, const int& line);

			/**
			 * Handles the results of a call to the NVML API.
			 * @param call The call.
			 * @param callResult The result of the call.
			 * @param file The file.
			 * @param line The line.
			 */
			static void handleAPICall(const std::string& call, const nvmlReturn_t& callResult, const std::string& file, const int& line);

			/**
			 * Gets the GPU with the specified device ID.
			 * @param id The device ID.
			 * @return The GPU.
			 */
			static std::shared_ptr<GPU> getGPU(const unsigned int& id);

			/**
			 * Gets all available GPUs.
			 * @return The GPUs.
			 */
			static std::vector<std::shared_ptr<GPU>> getGPUs();

			/**
			 * Gets the amount of GPUs.
			 * @return The amount of GPUs.
			 */
			static unsigned int getGPUCount();

			/**
			 * Makes this GPU the active GPU for future kernel launches.
			 */
			void makeActive() const;

			/**
			 * Gets the per Application clock rate.
			 * @return The Application clock rate.
			 */
			Utility::Units::Hertz getApplicationCoreClockRate() const;

			/**
			 * Sets the per Application clock rate.
			 * @param rate The Application clock rate.
			 */
			void setApplicationCoreClockRate(const Utility::Units::Hertz& rate);

			/**
			 * Resets the per Application clock rate.
			 */
			void resetApplicationCoreClockRate();

			/**
			 * Gets the per Application memory clock rate.
			 * @return The Application memory clock rate.
			 */
			Utility::Units::Hertz getApplicationMemoryClockRate() const;

			/**
			 * Sets the per Application memory clock rate.
			 * @param rate The Application memory clock rate.
			 */
			void setApplicationMemoryClockRate(const Utility::Units::Hertz& rate);

			/**
			 * Resets the per Application memory clock rate.
			 */
			void resetApplicationMemoryClockRate();

			/**
			 * Gets whether auto boosted clocks are enabled.
			 * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates to maximize performance as thermal limits allow.
			 * @return Whether auto boosted clocks are enabled.
			 */
			bool getAutoBoostedClocksEnabled() const;

			/**
			 * Gets whether auto boosted clocks are enabled by default.
			 * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates to maximize performance as thermal limits allow.
			 * @return Whether auto boosted clocks are enabled by default.
			 */
			bool getDefaultAutoBoostedClocksEnabled() const;

			/**
			 * Try to set the current state of Auto Boosted clocks on a device.
			 * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates to maximize performance as thermal limits allow. Auto Boosted clocks should be disabled if fixed clock rates are desired.
			 * @param enabled Whether to enable auto boosted clocks.
			 */
			void setAutoBoostedClocksEnabled(const bool& enabled);

			/**
			 * Returns the brand of the device.
			 * @return The brand.
			 */
			std::string getBrand() const;

			/**
			 * Gets the major version of the compute capability.
			 * @return The compute capability major version.
			 */
			unsigned int getComputeCapabilityMajorVersion() const;

			/**
			 * Gets the minor version of the compute capability.
			 * @return The compute capability minor version.
			 */
			unsigned int getComputeCapabilityMinorVersion() const;

			Utility::Units::Hertz getCoreClockRate() const final;

			Utility::Units::Hertz getCurrentMinimumCoreClockRate() const final;

			Utility::Units::Hertz getCurrentMaximumCoreClockRate() const final;

			void setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) final;

			/**
			 * Gets the supported core clock rates for the specified memory clock rate.
			 * @param memoryClockRate The memory clock rate.
			 * @return The supported core clock rates.
			 */
			std::vector<Utility::Units::Hertz> getSupportedCoreClockRates(const Utility::Units::Hertz& memoryClockRate) const;

			/**
			 * Gets the supported core clock rates for the current memory clock rate.
			 * @return The supported core clock rates.
			 */
			std::vector<Utility::Units::Hertz> getSupportedCoreClockRates() const;

			void resetCoreClockRate() final;

			Utility::Units::Hertz getMinimumCoreClockRate() const final;

			Utility::Units::Hertz getMaximumCoreClockRate() const final;

			Utility::Units::Percent getCoreUtilizationRate() const final;

			Utility::Units::Joule getEnergyConsumption() const final;

			Utility::Units::Watt getPowerConsumption() const final;

			/**
			 * Retrieves the power management limit associated with this device.
			 * @return The power limit.
			 */
			Utility::Units::Watt getPowerLimit() const;

			/**
			 * Retrieves default power management limit on this device, in milliwatts.
			 * @return The default power limit.
			 */
			Utility::Units::Watt getDefaultPowerLimit() const;

			/**
			 * Get the effective power limit that the driver enforces after taking into account all limiters.
			 * @return The enforced power limit.
			 */
			Utility::Units::Watt getEnforcedPowerLimit() const;

			/**
			 * @copydoc GPU::fanSpeed_
			 * @return The fan speed.
			 */
			Utility::Units::Percent getFanSpeed() const;

			Utility::Units::Percent getFanSpeed(const unsigned int& fan) const;

			/**
			 * @copydoc GPU::kernelBlockX_
			 * @return The kernel block X coordinate.
			 */
			int getKernelBlockX() const;

			/**
			 * @copydoc GPU::kernelBlockY_
			 * @return The kernel block Y coordinate.
			 */
			int getKernelBlockY() const;

			/**
			 * @copydoc GPU::kernelBlockZ_
			 * @return The kernel block Z coordinate.
			 */
			int getKernelBlockZ() const;

			/**
			 * @copydoc GPU::kernelContextID_
			 * @return The kernel context ID.
			 */
			unsigned int getKernelContextID() const;

			/**
			 * @copydoc GPU::kernelCorrelationID_
			 * @return The kernel correlation ID.
			 */
			unsigned int getKernelCorrelationID() const;

			/**
			 * @copydoc GPU::kernelDynamicSharedMemory_
			 * @return The kernel dynamic shared memory.
			 */
			Utility::Units::Byte getKernelDynamicSharedMemorySize() const;

			/**
			 * @copydoc GPU::kernelEndTimestamp_
			 * @return The kernel end timestamp.
			 */
			unsigned long getKernelEndTimestamp() const;

			/**
			 * @copydoc GPU::kernelGridX_
			 * @return The kernel grid X coordinate.
			 */
			int getKernelGridX() const;

			/**
			 * @copydoc GPU::kernelGridY_
			 * @return The kernel grid Y coordinate.
			 */
			int getKernelGridY() const;

			/**
			 * @copydoc GPU::kernelGridZ_
			 * @return The kernel grid Z coordinate.
			 */
			int getKernelGridZ() const;

			/**
			 * @copydoc GPU::kernelName_
			 * @return The kernel name.
			 */
			std::string getKernelName() const;

			/**
			 * @copydoc GPU::kernelStartTimestamp_
			 * @return The kernel start timestamp.
			 */
			unsigned long getKernelStartTimestamp() const;

			/**
			 * @copydoc GPU::kernelStaticSharedMemory_
			 * @return The kernel static shared memory.
			 */
			Utility::Units::Byte getKernelStaticSharedMemorySize() const;

			/**
			 * @copydoc GPU::kernelStreamID_
			 * @return The kernel stream ID.
			 */
			unsigned int getKernelStreamID() const;

			/**
			 * Retrieves the maximum clock speeds for the device.
			 * @return The maximum memory clock rate.
			 */
			Utility::Units::Hertz getMaximumMemoryClockRate() const;

			/**
			 * Gets the amount of memory.
			 * @return The memory clock.
			 */
			Utility::Units::Byte getMemorySize() const;

			/**
			 * Gets the amount of free memory.
			 * @return The memory clock.
			 */
			Utility::Units::Byte getMemoryFreeSize() const;

			/**
			 * Gets the amount of used memory.
			 * @return The memory clock.
			 */
			Utility::Units::Byte getMemoryUsedSize() const;

			/**
			 * @copydoc GPU::globalMemoryBandwidth_
			 * @return The global memory bandwidth.
			 */
			Utility::Units::Bandwidth getMemoryBandwidth() const;

			/**
			 * @copydoc Gets the clock rate of the memory.
			 * @return The memory clock.
			 */
			Utility::Units::Hertz getMemoryClockRate() const;

			/**
			 * Gets the synchronization mode currently set.
			 * @return The synchronization mode.
			 */
			SynchronizationMode getSynchronizationMode() const;

			/**
			 * Sets the synchronization mode.
			 * @param synchronizationMode The synchronization mode.
			 */
			void setSynchronizationMode(const SynchronizationMode& synchronizationMode);

			/**
			 * Gets the supported memory clock rates for the specified memory clock rate.
			 * @return The supported memory clock rates.
			 */
			std::vector<Utility::Units::Hertz> getSupportedMemoryClockRates() const;

			/**
			 * Gets the memory utilization rate.
			 * Percent of time over the past sample period during which global (device) memory was being read or written.
			 * @return The memory utilization rate.
			 */
			Utility::Units::Percent getMemoryUtilizationRate() const;

			/**
			 * @copydoc GPU::multiprocessorCount_
			 * @return The multiprocessor count.
			 */
			unsigned int getMultiprocessorCount() const;

			/**
			 * Gets the name of the GPU.
			 * @return The name.
			 */
			std::string getName() const;

			/**
			 * Retrieves the current PCIe link width.
			 * @return The PCIe link width.
			 */
			Utility::Units::Byte getPCIELinkWidth() const;

			/**
			 * Gets the clock rate of the streaming multiprocessors.
			 * @return The streaming multiprocessor clock.
			 */
			Utility::Units::Hertz getStreamingMultiprocessorClockRate() const;

			Utility::Units::Celsius getTemperature() const final;

			/**
			 * Resets any flags set for the device.
			 */
			void reset();
		};
	}
}