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

namespace EnergyManager {
	namespace Hardware {
		/**
		 * Represents a Central Processing Unit.
		 */
		class CPU : public Processor {
			/**
			 * Keeps track of CPUs.
			 */
			static std::map<uint32_t, std::shared_ptr<CPU>> cpus_;

			static std::chrono::system_clock::time_point lastProcCPUInfoValuesPerProcessorRetrieval;

			static std::map<unsigned int, std::map<std::string, std::string>> procCPUInfoValues;

			static std::mutex procCPUInfoValuesMutex_;

			static std::chrono::system_clock::time_point lastProcStatValuesRetrieval;

			static std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> procStatValues;

			static std::mutex procStatValuesMutex_;

			static std::mutex monitorThreadMutex_;

			/**
			 * Gets the current values of all CPUs.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<std::string, std::string>> getProcCPUInfoValuesPerProcessor();

			static std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> getProcCPUInfoValuesPerCPU();

			static std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> getProcStatValuesPerProcessor();

			static std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> getProcStatValuesPerCPU();

			/**
			 * The thread monitoring certain performance variables.
			 */
			std::thread monitorThread_;

			/**
			 * Whether the monitor thread should keep running.
			 */
			bool monitorThreadRunning_ = true;

			/**
			 * The last time at which the variables were polled.
			 */
			std::chrono::system_clock::time_point lastMonitorTimestamp_ = std::chrono::system_clock::now();

			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> lastProcStatValues_ = getProcStatValuesPerCPU();

			/**
			 * The last polled energy consumption.
			 */
			Utility::Units::Joule lastEnergyConsumption_ = 0;

			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> startProcStatValues_ = getProcStatValuesPerCPU();

			/**
			 * The starting energy consumption.
			 */
			Utility::Units::Joule startEnergyConsumption_ = 0;

			/**
			 * The ID of the device.
			 */
			unsigned int id_;

			/**
			 * The power consumption.
			 */
			Utility::Units::Watt powerConsumption_ = 0;

			std::map<unsigned int, Utility::Units::Percent> coreUtilizationRates_ = {};

			/**
			 * Creates a new CPU.
			 * @param id The ID of the device.
			 */
			CPU(const unsigned int& id);

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
			/**
			 * Gets the CPU with the specified ID.
			 * @param id The ID.
			 * @return The CPU.
			 */
			static std::shared_ptr<CPU> getCPU(const unsigned int& id);

			/**
			 * Gets the amount of CPUs.
			 * @return The amount of CPUs.
			 */
			static unsigned int getCPUCount();

			~CPU();

			Utility::Units::Hertz getCoreClockRate() const override;

			Utility::Units::Hertz getCoreClockRate(const unsigned int& core) const;

			void setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) override;

			void setCoreClockRate(const unsigned int& core, const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate);

			void resetCoreClockRate() override;

			void resetCoreClockRate(const unsigned int& core);

			Utility::Units::Percent getCoreUtilizationRate() const override;

			Utility::Units::Percent getCoreUtilizationRate(const unsigned int& core) const;

			Utility::Units::Joule getEnergyConsumption() const override;

			Utility::Units::Hertz getMaximumCoreClockRate() const override;

			Utility::Units::Hertz getMaximumCoreClockRate(const unsigned int& coreID) const;

			Utility::Units::Watt getPowerConsumption() const override;

			/**
			 * Get the core count.
			 * @return The amount of cores.
			 */
			unsigned int getCoreCount() const;

			std::chrono::system_clock::duration getUserTimespan() const;

			std::chrono::system_clock::duration getUserTimespan(const unsigned int& core) const;

			std::chrono::system_clock::duration getNiceTimespan() const;

			std::chrono::system_clock::duration getNiceTimespan(const unsigned int& core) const;

			std::chrono::system_clock::duration getSystemTimespan() const;

			std::chrono::system_clock::duration getSystemTimespan(const unsigned int& core) const;

			std::chrono::system_clock::duration getIdleTimespan() const;

			std::chrono::system_clock::duration getIdleTimespan(const unsigned int& core) const;

			std::chrono::system_clock::duration getIOWaitTimespan() const;

			std::chrono::system_clock::duration getIOWaitTimespan(const unsigned int& core) const;

			std::chrono::system_clock::duration getInterruptsTimespan() const;

			std::chrono::system_clock::duration getInterruptsTimespan(const unsigned int& core) const;

			std::chrono::system_clock::duration getSoftInterruptsTimespan() const;

			std::chrono::system_clock::duration getSoftInterruptsTimespan(const unsigned int& core) const;

			std::chrono::system_clock::duration getStealTimespan() const;

			std::chrono::system_clock::duration getStealTimespan(const unsigned int& core) const;

			std::chrono::system_clock::duration getGuestTimespan() const;

			std::chrono::system_clock::duration getGuestTimespan(const unsigned int& core) const;

			std::chrono::system_clock::duration getGuestNiceTimespan() const;

			std::chrono::system_clock::duration getGuestNiceTimespan(const unsigned int& core) const;

			Utility::Units::Celsius getTemperature() const override;
		};
	}
}