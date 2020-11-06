#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/Device.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Utility/Units/Byte.hpp"

#include <memory>
#include <vector>

namespace EnergyManager {
	namespace Hardware {
		/**
		 * Represents a system.
		 */
		class Node : public Device {
			/**
			 * The CPUs on the Node.
			 */
			std::vector<std::shared_ptr<Hardware::CPU>> cpus_;

			/**
			 * The GPUs on the Node.
			 */
			std::vector<std::shared_ptr<Hardware::GPU>> gpus_;

			/**
			 * The starting energy consumption.
			 */
			Utility::Units::Joule startEnergyConsumption_ = getEnergyConsumption();

			/**
			 * Creates a new Node.
			 */
			Node(std::vector<std::shared_ptr<Hardware::CPU>> cpus, std::vector<std::shared_ptr<Hardware::GPU>> gpus);

		public:
			/**
			 * Gets the current Node.
			 * @return The Node.
			 */
			static std::shared_ptr<Node> getNode();

			/**
			 * Gets the amount of memory on the Node.
			 * @return The memory size.
			 */
			Utility::Units::Byte getMemorySize() const;

			/**
			 * Gets the amount of free memory on the Node.
			 * @return The free memory size.
			 */
			Utility::Units::Byte getFreeMemorySize() const;

			/**
			 * Gets the amount of consumed memory on the Node.
			 * @return The used memory size.
			 */
			Utility::Units::Byte getUsedMemorySize() const;

			/**
			 * Gets the amount of shared memory on the Node.
			 * @return The shared memory size.
			 */
			Utility::Units::Byte getSharedMemorySize() const;

			/**
			 * Gets the amount of buffer memory on the Node.
			 * @return The buffer memory size.
			 */
			Utility::Units::Byte getBufferMemorySize() const;

			/**
			 * Gets the amount of swap memory on the Node.
			 * @return The swap memory size.
			 */
			Utility::Units::Byte getSwapMemorySize() const;

			/**
			 * Gets the amount of free swap memory on the Node.
			 * @return The free swap memory size.
			 */
			Utility::Units::Byte getFreeSwapMemorySize() const;

			/**
			 * Gets the amount of consumed swap memory on the Node.
			 * @return The used swap memory size.
			 */
			Utility::Units::Byte getUsedSwapMemorySize() const;

			/**
			 * Gets the amount of high memory on the Node.
			 * @return The high memory size.
			 */
			Utility::Units::Byte getHighMemorySize() const;

			/**
			 * Gets the amount of free high memory on the Node.
			 * @return The free high memory size.
			 */
			Utility::Units::Byte getFreeHighMemorySize() const;

			/**
			 * Gets the amount of consumed high memory on the Node.
			 * @return The used high memory size.
			 */
			Utility::Units::Byte getUsedHighMemorySize() const;

			/**
			 * Gets the amount of processes executing on the Node.
			 * @return The process count.
			 */
			unsigned int getProcessCount() const;

			Utility::Units::Joule getEnergyConsumption() const override;

			Utility::Units::Watt getPowerConsumption() const override;
		};
	}
}