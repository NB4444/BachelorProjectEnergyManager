#pragma once

#include "EnergyManager/Hardware/CentralProcessor.hpp"
#include "EnergyManager/Utility/CachedValue.hpp"
#include "EnergyManager/Utility/Logging/Loggable.hpp"
#include "EnergyManager/Utility/Loopable.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"
#include "EnergyManager/Utility/Units/Percent.hpp"
#include "EnergyManager/Utility/Units/Watt.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace EnergyManager {
	namespace Hardware {
		class Core;

		/**
		 * Represents a Central Processing Unit.
		 */
		class CPU
			: public CentralProcessor
			, protected Utility::Logging::Loggable {
			friend Core;

			/**
			 * The `/proc/stat` values at the start of execution.
			 */
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> startProcStatValues_ = getProcStatValuesPerCPU();

			/**
			 * The cores in the CPU.
			 */
			std::vector<std::shared_ptr<Core>> cores_;

			/**
			 * The starting energy consumption.
			 */
			Utility::Units::Joule startEnergyConsumption_ = 0;

			/**
			 * The power consumption.
			 */
			Utility::CachedValue<Utility::Units::Watt> powerConsumption;

			/**
			 * The last energy consumption recorded when calculating the power consumption.
			 */
			Utility::Units::Joule lastEnergyConsumption_;

			/**
			 * Creates a new CPU.
			 * @param id The ID of the device.
			 */
			explicit CPU(const unsigned int& id);

		protected:
			std::vector<std::string> generateHeaders() const override;

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

			Utility::Units::Percent getCoreUtilizationRate() final;

			Utility::Units::Joule getEnergyConsumption() final;

			Utility::Units::Hertz getMinimumCoreClockRate() const final;

			Utility::Units::Hertz getMaximumCoreClockRate() const final;

			Utility::Units::Watt getPowerConsumption() final;

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
		};
	}
}