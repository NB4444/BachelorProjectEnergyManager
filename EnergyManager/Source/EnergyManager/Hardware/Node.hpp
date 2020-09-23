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
		 * Represents a Graphics Processing Unit.
		 */
		class Node : public Device {
			std::vector<std::shared_ptr<Hardware::CPU>> cpus_;

			std::vector<std::shared_ptr<Hardware::GPU>> gpus_;

			/**
			 * The starting energy consumption.
			 */
			Utility::Units::Joule startEnergyConsumption_ = 0;

			/**
			 * Creates a new Node.
			 */
			Node(std::vector<std::shared_ptr<Hardware::CPU>> cpus, std::vector<std::shared_ptr<Hardware::GPU>> gpus);

		public:
			/**
			 * Gets the Node with the specified device ID.
			 * @param id The device ID.
			 * @return The Node.
			 */
			static std::shared_ptr<Node> getNode();

			Utility::Units::Byte getMemorySize() const;

			Utility::Units::Byte getFreeMemorySize() const;

			Utility::Units::Byte getUsedMemorySize() const;

			Utility::Units::Byte getSharedMemorySize() const;

			Utility::Units::Byte getBufferMemorySize() const;

			Utility::Units::Byte getSwapMemorySize() const;

			Utility::Units::Byte getFreeSwapMemorySize() const;

			Utility::Units::Byte getUsedSwapMemorySize() const;

			Utility::Units::Byte getHighMemorySize() const;

			Utility::Units::Byte getFreeHighMemorySize() const;

			Utility::Units::Byte getUsedHighMemorySize() const;

			unsigned int getProcessCount() const;

			Utility::Units::Joule getEnergyConsumption() const override;

			Utility::Units::Watt getPowerConsumption() const override;
		};
	}
}