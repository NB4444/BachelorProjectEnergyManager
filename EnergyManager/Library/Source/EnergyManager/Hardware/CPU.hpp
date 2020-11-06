#pragma once

#include "EnergyManager/Hardware/Processor.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"
#include "EnergyManager/Utility/Units/Percent.hpp"
#include "EnergyManager/Utility/Units/Watt.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace EnergyManager {
	namespace Hardware {
		/**
		 * Represents a Central Processing Unit.
		 */
		class CPU : public Processor {
			/**
			 * The mutex used to access variables that are recorded by the monitor thread.
			 */
			static std::mutex monitorThreadMutex_;

			/**
			 * Gets the current `/proc/cpuinfo` values of all available CPU processors.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<std::string, std::string>> getProcCPUInfoValuesPerProcessor();

			/**
			 * Gets the current `/proc/cpuinfo` values of all available CPUs.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> getProcCPUInfoValuesPerCPU();

			/**
			 * Gets the current `/proc/stat` values of all available CPU processors.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> getProcStatValuesPerProcessor();

			/**
			 * Gets the current `/proc/stat` values of all available CPUs.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> getProcStatValuesPerCPU();

			/**
			 * The ID of the device.
			 */
			unsigned int id_;

			/**
			 * The thread monitoring certain performance variables.
			 */
			std::thread monitorThread_;

			/**
			 * Whether the monitor thread should keep running.
			 */
			bool monitorThreadRunning_ = true;

			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> startProcStatValues_ = getProcStatValuesPerCPU();

			/**
			 * The starting energy consumption.
			 */
			Utility::Units::Joule startEnergyConsumption_ = 0;

			/**
			 * The power consumption.
			 */
			Utility::Units::Watt powerConsumption_ = 0;

			/**
			 * The utilization rates per core.
			 */
			std::map<unsigned int, Utility::Units::Percent> coreUtilizationRates_ = {};

			/**
			 * Creates a new CPU.
			 * @param id The ID of the device.
			 */
			explicit CPU(const unsigned int& id);

			/**
			 * Converts a core ID to a processor ID for use with system commands.
			 * @param core The core ID.
			 * @return The processor ID.
			 */
			unsigned int getProcessorID(const unsigned int& core) const;

			/**
			 * Gets a `/proc/stat` timespan.
			 * @param core The core.
			 * @param name The name of the timespan.
			 * @return The timespan.
			 */
			std::chrono::system_clock::duration getProcStatTimespan(const unsigned int& core, const std::string& name) const;

		public:
			static std::vector<std::shared_ptr<Hardware::CPU>> parseCPUs(const std::string& cpuString);

			/**
			 * Gets the CPU with the specified ID.
			 * @param id The ID.
			 * @return The CPU.
			 */
			static std::shared_ptr<CPU> getCPU(const unsigned int& id);

			/**
			 * Gets all available CPUs.
			 * @return The CPUs.
			 */
			static std::vector<std::shared_ptr<CPU>> getCPUs();

			/**
			 * Gets the amount of CPUs.
			 * @return The amount of CPUs.
			 */
			static unsigned int getCPUCount();

			/**
			 * Makes sure all threads are stopped when the application stops.
			 */
			~CPU();

			/**
			 * Gets the ID of the CPU.
			 * @return The ID.
			 */
			unsigned int getID() const;

			Utility::Units::Hertz getCoreClockRate() const override;

			Utility::Units::Hertz getCurrentMinimumCoreClockRate() const override;

			Utility::Units::Hertz getCurrentMaximumCoreClockRate() const override;

			/**
			 * Gets the clock rate of the specified core.
			 * @param core The core.
			 * @return The clock rate.
			 */
			Utility::Units::Hertz getCoreClockRate(const unsigned int& core) const;

			/**
			 * Gets the minimum clock rate of the specified core.
			 * @param core The core.
			 * @return The clock rate.
			 */
			Utility::Units::Hertz getCurrentMinimumCoreClockRate(const unsigned int& core) const;

			/**
			 * Gets the maximum clock rate of the specified core.
			 * @param core The core.
			 * @return The clock rate.
			 */
			Utility::Units::Hertz getCurrentMaximumCoreClockRate(const unsigned int& core) const;

			void setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) override;

			/**
			 * Sets the clock rate boundaries of the specified core.
			 * @param core The core.
			 * @param minimumRate The minimum clock rate.
			 * @param maximumRate The maximum clock rate.
			 */
			void setCoreClockRate(const unsigned int& core, const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate);

			void resetCoreClockRate() override;

			/**
			 * Resets the clock rates of the specified core
			 * @param core The core.
			 */
			void resetCoreClockRate(const unsigned int& core);

			/**
			 * Get the core count.
			 * @return The amount of cores.
			 */
			unsigned int getCoreCount() const;

			Utility::Units::Percent getCoreUtilizationRate() const override;

			/**
			 * Gets the utilization rate of the specified core.
			 * @param core The core.
			 * @return The utilization rate.
			 */
			Utility::Units::Percent getCoreUtilizationRate(const unsigned int& core) const;

			Utility::Units::Joule getEnergyConsumption() const override;

			Utility::Units::Hertz getMinimumCoreClockRate() const override;

			Utility::Units::Hertz getMaximumCoreClockRate() const override;

			/**
			 * Gets the maximum clock rate of the specified core.
			 * @param core The core.
			 * @return The maximum clock rate.
			 */
			Utility::Units::Hertz getMaximumCoreClockRate(const unsigned int& core) const;

			/**
			 * Gets the minimum clock rate of the specified core.
			 * @param core The core.
			 * @return The minimum clock rate.
			 */
			Utility::Units::Hertz getMinimumCoreClockRate(const unsigned int& core) const;

			Utility::Units::Watt getPowerConsumption() const override;

			/**
			 * Gets the amount of time spent on user level processes.
			 * @return The user timespan.
			 */
			std::chrono::system_clock::duration getUserTimespan() const;

			/**
			 * Gets the amount of time spent on user level processes by the specified core.
			 * @param core The core.
			 * @return The user timespan.
			 */
			std::chrono::system_clock::duration getUserTimespan(const unsigned int& core) const;

			/**
			 * Gets the amount of time spent on user level processes with a positive nice value.
			 * @return The nice timespan.
			 */
			std::chrono::system_clock::duration getNiceTimespan() const;

			/**
			 * Gets the amount of time spent on user level processes with a positive nice value by the specified core.
			 * @param core The core.
			 * @return The nice timespan.
			 */
			std::chrono::system_clock::duration getNiceTimespan(const unsigned int& core) const;

			/**
			 * Gets the amount of time spent on system level processes.
			 * @return The system timespan.
			 */
			std::chrono::system_clock::duration getSystemTimespan() const;

			/**
			 * Gets the amount of time spent on system level processes by the specified core.
			 * @param core The core.
			 * @return The system timespan.
			 */
			std::chrono::system_clock::duration getSystemTimespan(const unsigned int& core) const;

			/**
			 * Gets the amount of time spent idle.
			 * @return The idle timespan.
			 */
			std::chrono::system_clock::duration getIdleTimespan() const;

			/**
			 * Gets the amount of time spent idle by the specified core.
			 * @param core The core.
			 * @return The idle timespan.
			 */
			std::chrono::system_clock::duration getIdleTimespan(const unsigned int& core) const;

			/**
			 * Gets the amount of time spent waiting for IO operations.
			 * @return The IO wait timespan.
			 */
			std::chrono::system_clock::duration getIOWaitTimespan() const;

			/**
			 * Gets the amount of time spent waiting for IO operations by the specified core.
			 * @param core The core.
			 * @return The IO wait timespan.
			 */
			std::chrono::system_clock::duration getIOWaitTimespan(const unsigned int& core) const;

			/**
			 * Gets the amount of time spent on interrupts.
			 * @return The interrupts timespan.
			 */
			std::chrono::system_clock::duration getInterruptsTimespan() const;

			/**
			 * Gets the amount of time spent on interrupts by the specified core.
			 * @param core The core.
			 * @return The interrupts timespan.
			 */
			std::chrono::system_clock::duration getInterruptsTimespan(const unsigned int& core) const;

			/**
			 * Gets the power scaling driver that is in use.
			 * @param core The core.
			 * @return The power scaling driver.
			 */
			std::string getPowerScalingDriver(const unsigned int& core) const;

			/**
			 * Gets the amount of time spent on soft interrupts.
			 * @return The soft interrupts timespan.
			 */
			std::chrono::system_clock::duration getSoftInterruptsTimespan() const;

			/**
			 * Gets the amount of time spent on soft interrupts by the specified core.
			 * @param core The core.
			 * @return The soft interrupts timespan.
			 */
			std::chrono::system_clock::duration getSoftInterruptsTimespan(const unsigned int& core) const;

			/**
			 * Gets the amount of time waiting on the host CPU in a virtualized environment.
			 * @return The steal timespan.
			 */
			std::chrono::system_clock::duration getStealTimespan() const;

			/**
			 * Gets the amount of time waiting on the host CPU in a virtualized environment by the specified core.
			 * @param core The core.
			 * @return The steal timespan.
			 */
			std::chrono::system_clock::duration getStealTimespan(const unsigned int& core) const;

			/**
			 * Gets the amount of time spent on processes in a guest virtualization environment.
			 * @return The guest timespan.
			 */
			std::chrono::system_clock::duration getGuestTimespan() const;

			/**
			 * Gets the amount of time spent on processes in a guest virtualization environment by the specified core.
			 * @param core The core.
			 * @return The guest timespan.
			 */
			std::chrono::system_clock::duration getGuestTimespan(const unsigned int& core) const;

			/**
			 * Gets the amount of time spent on user level processes with a positive nice value in a guest virtualization environment.
			 * @return The nice timespan.
			 */
			std::chrono::system_clock::duration getGuestNiceTimespan() const;

			/**
			 * Gets the amount of time spent on user level processes with a positive nice value in a guest virtualization environment by the specified core.
			 * @param core The core.
			 * @return The nice timespan.
			 */
			std::chrono::system_clock::duration getGuestNiceTimespan(const unsigned int& core) const;

			Utility::Units::Celsius getTemperature() const override;

			/**
			 * Determines if turbo is enabled.
			 * @return Whether turbo is enabled.
			 */
			bool getTurboEnabled() const;

			/**
			 * Enables or disables turbo.
			 * @param turbo Whether to enable turbo.
			 */
			void setTurboEnabled(const bool& turbo) const;
		};
	}
}