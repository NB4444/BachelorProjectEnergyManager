#pragma once

#include "EnergyManager/Hardware/Processor.hpp"

#include <map>
#include <memory>
#include <string>

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
			 * The ID of the device.
			 */
			unsigned int id_;

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

			unsigned long getCoreClockRate() const override;

			unsigned long getMaximumCoreClockRate() const override;
		};
	}
}