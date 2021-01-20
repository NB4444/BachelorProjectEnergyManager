#pragma once

#include "EnergyManager/Hardware/CentralProcessor.hpp"
#include "EnergyManager/Utility/Logging/Loggable.hpp"

#include <memory>

namespace EnergyManager {
	namespace Hardware {
		class CPU;

		/**
		 * Represents a CPU core.
		 */
		class Core
			: public CentralProcessor
			, protected Utility::Logging::Loggable {
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
			 * The last update of the utilization rate.
			 */
			std::chrono::system_clock::time_point lastUtilizationRateUpdate_ = std::chrono::system_clock::now();

			/**
			 * The proc stat values when the utilization rate was last updated.
			 */
			std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> lastUtilizationRateUpdateProcStatValues_;

			/**
			 * Gets a `/proc/stat` timespan.
			 * @param name The name of the timespan.
			 * @return The timespan.
			 */
			std::chrono::system_clock::duration getProcStatTimespan(const std::string& name) const;

			/**
			 * Gets the Core with the specified CPU and local ID.
			 * @param cpuID The CPU ID.
			 * @param id The ID.
			 * @return The core.
			 */
			static std::shared_ptr<Core> getCore(CPU* cpu, const unsigned int& id);

		protected:
			/**
			 * Creates a new Core.
			 * @param cpu The CPU that the core belongs to.
			 * @param id The ID of the core.
			 * @param coreID The ID of the core in the current CPU.
			 */
			explicit Core(CPU* cpu, const unsigned int& id, const unsigned int& coreID);

			std::vector<std::string> generateHeaders() const override;

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
			 * Converts the unique core ID to an ID local to the current CPU.
			 * @return The core ID.
			 */
			unsigned int getCoreID() const;

			Utility::Units::Hertz getCoreClockRate() const final;

			Utility::Units::Hertz getCurrentMinimumCoreClockRate() const final;

			Utility::Units::Hertz getCurrentMaximumCoreClockRate() const final;

			void setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) final;

			void resetCoreClockRate() final;

			Utility::Units::Percent getCoreUtilizationRate() final;

			Utility::Units::Hertz getMinimumCoreClockRate() const final;

			Utility::Units::Hertz getMaximumCoreClockRate() const final;

			Utility::Units::Joule getEnergyConsumption() final;

			Utility::Units::Watt getPowerConsumption() final;

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
	}
}