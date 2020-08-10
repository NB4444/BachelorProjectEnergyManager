#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <string>

namespace Profiling {
	/**
	 * Monitors a set of variables and keeps track of their values over time.
	 */
	class Monitor {
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
		 * @param variableProducers
		 */
		Monitor() = default;

		/**
		 * Polls all variables and stores their current values.
		 * @return The variables at the current point in time.
		 */
		std::map<std::string, std::string> poll();
	};
}