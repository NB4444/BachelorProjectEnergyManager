#pragma once

#include "EnergyManager/Hardware/CentralProcessor.hpp"
#include "EnergyManager/Utility/Loopable.hpp"
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
		class CPU
			: public CentralProcessor
			, private Utility::Loopable {
		public:
			class Core;

		private:
			friend Core;

			/**
			 * The mutex that protects the monitored resources.
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
			 * The cores of the current CPU.
			 */
			std::vector<std::shared_ptr<Core>> cores_;

			/**
			 * Record the last time at which the variables were polled
			 */
			std::chrono::system_clock::time_point lastMonitorTimestamp_ = std::chrono::system_clock::now();

			/**
			 * Record the last `/proc/stat` values
			 */
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> lastProcStatValues_ = getProcStatValuesPerCPU();

			/**
			 * Record the last polled energy consumption
			 */
			Utility::Units::Joule lastEnergyConsumption_ = 0;

			/**
			 * The `/proc/stat` values at the start of execution.
			 */
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

		protected:
			void onLoop() override;

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
			 * Gets the cores in the CPU.
			 * @return The cores.
			 */
			std::vector<std::shared_ptr<Core>> getCores() const;

			Utility::Units::Hertz getCoreClockRate() const final;

			Utility::Units::Hertz getCurrentMinimumCoreClockRate() const final;

			Utility::Units::Hertz getCurrentMaximumCoreClockRate() const final;

			void setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) final;

			void resetCoreClockRate() final;

			Utility::Units::Percent getCoreUtilizationRate() const final;

			Utility::Units::Joule getEnergyConsumption() const final;

			Utility::Units::Hertz getMinimumCoreClockRate() const final;

			Utility::Units::Hertz getMaximumCoreClockRate() const final;

			Utility::Units::Watt getPowerConsumption() const final;

			std::chrono::system_clock::duration getUserTimespan() const final;

			std::chrono::system_clock::duration getNiceTimespan() const final;

			std::chrono::system_clock::duration getSystemTimespan() const final;

			std::chrono::system_clock::duration getIdleTimespan() const final;

			std::chrono::system_clock::duration getIOWaitTimespan() const final;

			std::chrono::system_clock::duration getInterruptsTimespan() const final;

			std::chrono::system_clock::duration getSoftInterruptsTimespan() const final;

			std::chrono::system_clock::duration getStealTimespan() const final;

			std::chrono::system_clock::duration getGuestTimespan() const final;

			std::chrono::system_clock::duration getGuestNiceTimespan() const final;

			Utility::Units::Celsius getTemperature() const final;

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

			/**
			 * Represents a CPU core.
			 */
			class Core : public CentralProcessor {
				friend CPU;

				/**
				 * The CPU that the core belongs to.
				 */
				CPU* cpu_;

				/**
				 * The core ID.
				 */
				unsigned int coreID_;

				/**
				 * Creates a new Core.
				 * @param cpu The CPU that the core belongs to.
				 * @param id The ID of the core.
				 * @param coreID The ID of the core in the current CPU.
				 */
				explicit Core(CPU* cpu, const unsigned int& id, const unsigned int& coreID);

				/**
				 * Gets a `/proc/stat` timespan.
				 * @param name The name of the timespan.
				 * @return The timespan.
				 */
				std::chrono::system_clock::duration getProcStatTimespan(const std::string& name) const;

			public:
				/**
				 * Gets the Core with the specified ID.
				 * @param id The ID.
				 * @return The Core.
				 */
				static std::shared_ptr<Core> getCore(const unsigned int& id);

				/**
				 * Gets the CPU that the core belongs to.
				 * @return The CPU.
				 */
				std::shared_ptr<CPU> getCPU() const;

				/**
				 * Converts a core ID to a processor ID for use with system commands.
				 * @return The processor ID.
				 */
				unsigned int getCoreID() const;

				Utility::Units::Hertz getCoreClockRate() const final;

				Utility::Units::Hertz getCurrentMinimumCoreClockRate() const final;

				Utility::Units::Hertz getCurrentMaximumCoreClockRate() const final;

				void setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) final;

				void resetCoreClockRate() final;

				Utility::Units::Percent getCoreUtilizationRate() const final;

				Utility::Units::Hertz getMinimumCoreClockRate() const final;

				Utility::Units::Hertz getMaximumCoreClockRate() const final;

				Utility::Units::Joule getEnergyConsumption() const final;

				Utility::Units::Watt getPowerConsumption() const final;

				Utility::Units::Celsius getTemperature() const final;

				/**
				 * Gets the amount of time spent on user level processes.
				 * @return The user timespan.
				 */
				std::chrono::system_clock::duration getUserTimespan() const final;

				/**
				 * Gets the amount of time spent on user level processes with a positive nice value.
				 * @return The nice timespan.
				 */
				std::chrono::system_clock::duration getNiceTimespan() const final;

				/**
				 * Gets the amount of time spent on system level processes.
				 * @return The system timespan.
				 */
				std::chrono::system_clock::duration getSystemTimespan() const final;

				/**
				 * Gets the amount of time spent idle.
				 * @return The idle timespan.
				 */
				std::chrono::system_clock::duration getIdleTimespan() const final;

				/**
				 * Gets the amount of time spent waiting for IO operations.
				 * @return The IO wait timespan.
				 */
				std::chrono::system_clock::duration getIOWaitTimespan() const final;

				/**
				 * Gets the amount of time spent on interrupts.
				 * @return The interrupts timespan.
				 */
				std::chrono::system_clock::duration getInterruptsTimespan() const final;

				/**
				 * Gets the amount of time spent on soft interrupts by the specified core.
				 * @return The soft interrupts timespan.
				 */
				std::chrono::system_clock::duration getSoftInterruptsTimespan() const final;

				/**
				 * Gets the amount of time waiting on the host CPU in a virtualized environment by the specified core.
				 * @return The steal timespan.
				 */
				std::chrono::system_clock::duration getStealTimespan() const final;

				/**
				 * Gets the amount of time spent on processes in a guest virtualization environment by the specified core.
				 * @return The guest timespan.
				 */
				std::chrono::system_clock::duration getGuestTimespan() const final;

				/**
				 * Gets the amount of time spent on user level processes with a positive nice value in a guest virtualization environment by the specified core.
				 * @return The nice timespan.
				 */
				std::chrono::system_clock::duration getGuestNiceTimespan() const final;

				/**
				 * Gets the power scaling driver that is in use.
				 * @return The power scaling driver.
				 */
				std::string getPowerScalingDriver() const;
			};
		};
	}
}