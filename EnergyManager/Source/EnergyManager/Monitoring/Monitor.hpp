#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace EnergyManager {
	namespace Monitoring {
		/**
		 * Monitors a set of variables and keeps track of their values over time.
		 */
		class Monitor {
			using Parser = std::function<std::shared_ptr<Monitor>(const std::string& name, const std::map<std::string, std::string>& parameters)>;

			static std::vector<Parser> parsers_;

			/**
			 * The name of the Monitor.
			 */
			std::string name_;

			/**
			 * Whether the Monitor is running.
			 */
			bool isRunning_ = false;

			/**
			 * The start timestamp.
			 */
			std::chrono::system_clock::time_point startTimestamp_;

			/**
			 * The last polling timestamp.
			 */
			std::chrono::system_clock::time_point lastPollTimestamp_;

			/**
			 * The variables at different points in time.
			 */
			std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> variableValues_;

		protected:
			static void addParser(const Parser& parser);

			/**
			 * Executes when the monitor is polled.
			 * @return The variables at the current point in time.
			 */
			virtual std::map<std::string, std::string> onPoll();

		public:
			static std::shared_ptr<Monitor> parse(const std::string& name, const std::map<std::string, std::string>& parameters);

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
			 * Gets the start timestamp.
			 * @return The start timestamp.
			 */
			std::chrono::system_clock::time_point getStartTimestamp() const;

			/**
			 * Gets the total runtime.
			 * @return The runtime.
			 */
			std::chrono::system_clock::duration getRuntime() const;

			/**
			 * Gets the timestamp of the last poll operation.
			 * @return The last poll timestamp.
			 */
			std::chrono::system_clock::time_point getLastPollTimestamp() const;

			/**
			 * Gets the time since the last poll operation.
			 * @return The time since the last poll operation.
			 */
			std::chrono::system_clock::duration getTimeSinceLastPoll() const;

			bool isRunning() const;

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
			void run(const std::chrono::system_clock::duration& interval);

			/**
			 * Stops the Monitor if it is running.
			 */
			void stop();
		};
	}
}