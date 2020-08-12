#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <string>

namespace EnergyManager {
	namespace Profiling {
		/**
		 * Monitors a set of variables and keeps track of their values over time.
		 */
		class Monitor {
			/**
			 * The name of the Monitor.
			 */
			std::string name_;

			/**
			 * Whether the Monitor is running.
			 */
			bool running_ = false;

			/**
			 * The variables at different points in time.
			 */
			std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> variableValues_;

		protected:
			/**
			 * Executes when the monitor is polled.
			 * @return The variables at the current point in time.
			 */
			virtual std::map<std::string, std::string> onPoll();

		public:
			/**
			 * Creates a new Monitor.
			 * @param name The name of the Monitor.
			 */
			Monitor(std::string name);

			/**
			 * Gets the name of the Monitor.
			 * @return The name.
			 */
			std::string getName() const;

			/**
			 * Gets the variables at different points in time.
			 * @return The variable values.
			 */
			std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> getVariableValues() const;

			/**
			 * Polls all variables and potentially stores their current values.
			 * @param save Whether to save the polled variables.
			 * @return The variables at the current point in time.
			 */
			std::map<std::string, std::string> poll(const bool& save);

			/**
			 * Runs the Monitor using a specified polling interval.
			 * @param interval The polling interval in seconds.
			 */
			void run(const std::chrono::seconds& interval);

			/**
			 * Stops the Monitor if it is running.
			 */
			void stop();
		};
	}
}