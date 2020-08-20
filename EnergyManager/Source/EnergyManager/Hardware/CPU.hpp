#pragma once

#include "EnergyManager/Hardware/Processor.hpp"

#include <map>
#include <memory>
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

			/**
			 * Gets the current values of all CPUs.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<std::string, std::string>> getProcCPUInfoValues();

			/**
			 * The thread monitoring certain performance variables.
			 */
			std::thread monitorThread_;

			/**
			 * Whether the monitor thread should keep running.
			 */
			bool monitorThreadRunning_ = true;

			/**
			 * The last polled energy consumption in Joules.
			 */
			float lastEnergyConsumption_ = 0;

			/**
			 * The starting energy consumption.
			 */
			float startEnergyConsumption_ = 0;

			/**
			 * Whether the start energy consumption was measured.
			 */
			bool startEnergyConsumptionMeasured_ = false;

			/**
			 * The last time at which the energy consumption was polled.
			 */
			std::chrono::system_clock::time_point lastEnergyConsumptionPollTimestamp_ = std::chrono::system_clock::now();

			/**
			 * The ID of the device.
			 */
			unsigned int id_;

			/**
			 * The power consumption in Watts.
			 */
			float powerConsumption_ = 0;

			/**
			 * Creates a new CPU.
			 * @param id The ID of the device.
			 */
			CPU(const unsigned int& id);

		public:
			/**
			 * Gets the CPU with the specified ID.
			 * @param id The ID.
			 * @return The CPU.
			 */
			static std::shared_ptr<CPU> getCPU(const unsigned int& id);

			~CPU();

			unsigned long getCoreClockRate() const override;

			void setCoreClockRate(unsigned long& rate) override;

			float getEnergyConsumption() const override;

			unsigned long getMaximumCoreClockRate() const override;

			float getPowerConsumption() const override;
		};
	}
}