#pragma once

#include "EnergyManager/Hardware/Processor.hpp"

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <mutex>

namespace EnergyManager {
	namespace Hardware {
		/**
		 * Represents a Central Processing Unit.
		 */
		class CPU :
			public Processor {
				/**
				 * Keeps track of CPUs.
				 */
				static std::map<uint32_t, std::shared_ptr<CPU>> cpus_;

				/**
				 * Gets the current values of all CPUs.
				 * @return The current values.
				 */
				static std::map<unsigned int, std::map<std::string, std::string>> getProcCPUInfoValuesPerProcessor();

				static std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> getProcCPUInfoValuesPerCPU();

				static std::map<unsigned int, std::map<std::string, double>> getProcStatValuesPerProcessor();

				static std::map<unsigned int, std::map<unsigned int, std::map<std::string, double>>> getProcStatValuesPerCPU();

				static std::mutex monitorThreadMutex_;

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

				std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> lastProcCPUInfoValues_ = getProcCPUInfoValuesPerCPU();

				std::map<unsigned int, std::map<unsigned int, std::map<std::string, double>>> lastProcStatValues_ = getProcStatValuesPerCPU();

				/**
				 * The last polled energy consumption in Joules.
				 */
				float lastEnergyConsumption_ = 0;

				std::map<unsigned int, std::map<unsigned int, std::map<std::string, double>>> startProcStatValues_ = getProcStatValuesPerCPU();

				/**
				 * The starting energy consumption.
				 */
				float startEnergyConsumption_ = 0;

				/**
				 * The ID of the device.
				 */
				unsigned int id_;

				/**
				 * The power consumption in Watts.
				 */
				float powerConsumption_ = 0;

				std::map<unsigned int, float> coreUtilizationRates_ = {};

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
				double getProcStatTimespan(const unsigned int& core, const std::string& name) const;

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

				unsigned long getCoreClockRate() const override;

				unsigned long getCoreClockRate(const unsigned int& coreID) const;

				void setCoreClockRate(unsigned long& rate) override;

				float getCoreUtilizationRate() const override;

				float getCoreUtilizationRate(const unsigned int& core) const;

				float getEnergyConsumption() const override;

				unsigned long getMaximumCoreClockRate() const override;

				unsigned long getMaximumCoreClockRate(const unsigned int& coreID) const;

				float getPowerConsumption() const override;

				/**
				 * Get the core count.
				 * @return The amount of cores.
				 */
				unsigned int getCoreCount() const;

				double getUserTimespan() const;

				double getUserTimespan(const unsigned int& core) const;

				double getNiceTimespan() const;

				double getNiceTimespan(const unsigned int& core) const;

				double getSystemTimespan() const;

				double getSystemTimespan(const unsigned int& core) const;

				double getIdleTimespan() const;

				double getIdleTimespan(const unsigned int& core) const;

				double getIOWaitTimespan() const;

				double getIOWaitTimespan(const unsigned int& core) const;

				double getInterruptsTimespan() const;

				double getInterruptsTimespan(const unsigned int& core) const;

				double getSoftInterruptsTimespan() const;

				double getSoftInterruptsTimespan(const unsigned int& core) const;

				double getStealTimespan() const;

				double getStealTimespan(const unsigned int& core) const;

				double getGuestTimespan() const;

				double getGuestTimespan(const unsigned int& core) const;

				double getGuestNiceTimespan() const;

				double getGuestNiceTimespan(const unsigned int& core) const;
		};
	}
}