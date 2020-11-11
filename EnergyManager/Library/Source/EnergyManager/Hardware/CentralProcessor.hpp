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
		class CentralProcessor : public Processor {
			using Processor::Processor;
			/**
			 * Gets the current `/proc/CentralProcessorinfo` values of all available CentralProcessor processors.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<std::string, std::string>> getProcCentralProcessorInfoValuesPerProcessor();

			/**
			 * Gets the current `/proc/CentralProcessorinfo` values of all available CentralProcessors.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> getProcCentralProcessorInfoValuesPerCentralProcessor();

			/**
			 * Gets the current `/proc/stat` values of all available CentralProcessor processors.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> getProcStatValuesPerProcessor();

			/**
			 * Gets the current `/proc/stat` values of all available CentralProcessors.
			 * @return The current values.
			 */
			static std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> getProcStatValuesPerCentralProcessor();

		public:
			/**
			 * Gets the amount of time spent on user level processes.
			 * @return The user timespan.
			 */
			virtual std::chrono::system_clock::duration getUserTimespan() const = 0;

			/**
			 * Gets the amount of time spent on user level processes with a positive nice value.
			 * @return The nice timespan.
			 */
			virtual std::chrono::system_clock::duration getNiceTimespan() const = 0;

			/**
			 * Gets the amount of time spent on system level processes.
			 * @return The system timespan.
			 */
			virtual std::chrono::system_clock::duration getSystemTimespan() const = 0;

			/**
			 * Gets the amount of time spent idle.
			 * @return The idle timespan.
			 */
			virtual std::chrono::system_clock::duration getIdleTimespan() const = 0;

			/**
			 * Gets the amount of time spent waiting for IO operations.
			 * @return The IO wait timespan.
			 */
			virtual std::chrono::system_clock::duration getIOWaitTimespan() const = 0;

			/**
			 * Gets the amount of time spent on interrupts.
			 * @return The interrupts timespan.
			 */
			virtual std::chrono::system_clock::duration getInterruptsTimespan() const = 0;

			/**
			 * Gets the amount of time spent on soft interrupts.
			 * @return The soft interrupts timespan.
			 */
			virtual std::chrono::system_clock::duration getSoftInterruptsTimespan() const = 0;

			/**
			 * Gets the amount of time waiting on the host CentralProcessor in a virtualized environment.
			 * @return The steal timespan.
			 */
			virtual std::chrono::system_clock::duration getStealTimespan() const = 0;

			/**
			 * Gets the amount of time spent on processes in a guest virtualization environment.
			 * @return The guest timespan.
			 */
			virtual std::chrono::system_clock::duration getGuestTimespan() const = 0;

			/**
			 * Gets the amount of time spent on user level processes with a positive nice value in a guest virtualization environment.
			 * @return The nice timespan.
			 */
			virtual std::chrono::system_clock::duration getGuestNiceTimespan() const = 0;
		};
	}
}