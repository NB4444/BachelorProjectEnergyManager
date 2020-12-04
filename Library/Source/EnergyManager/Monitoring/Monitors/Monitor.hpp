#pragma once

#include "EnergyManager/Utility/Logging/Loggable.hpp"
#include "EnergyManager/Utility/Loopable.hpp"

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a set of variables and keeps track of their values over time.
			 */
			class Monitor : public Utility::Loopable {
				/**
				 * The name of the Monitor.
				 */
				std::string name_;

				/**
				 * The variables at different points in time.
				 */
				std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> variableValues_;

			protected:
				std::vector<std::string> generateHeaders() const override;

				void onLoop() final;

				/**
				 * Executes when the monitor is polled.
				 * @return The variables at the current point in time.
				 */
				virtual std::map<std::string, std::string> onPoll();

				/**
				 * Executes when the monitor is reset.
				 */
				virtual void onReset();

				/**
				 * Sets a variable value.
				 * @param timestamp The timestamp to use.
				 * @param name The name of the variable.
				 * @param value The value.
				 */
				void setVariable(const std::chrono::system_clock::time_point& timestamp, const std::string& name, const std::string& value);

			public:
				/**
				 * Gets a list of monitors for all available devices.
				 * @param applicationMonitorInterval The interval at which to run the ApplicationMonitor.
				 * @param nodeMonitorInterval The interval at which to run the NodeMonitor.
				 * @param cpuMonitorInterval The interval at which to run the CPUMonitor.
				 * @param cpuCoreMonitorInterval The interval at which to run the CPUCoreMonitor.
				 * @param gpuMonitorInterval The interval at which to run the GPUMonitor.
				 * @return The monitors.
				 */
				static std::vector<std::shared_ptr<Monitor>> getMonitorsForAllDevices(
					const std::chrono::system_clock::duration& applicationMonitorInterval = std::chrono::milliseconds(100),
					const std::chrono::system_clock::duration& nodeMonitorInterval = std::chrono::milliseconds(100),
					const std::chrono::system_clock::duration& cpuMonitorInterval = std::chrono::milliseconds(100),
					const std::chrono::system_clock::duration& cpuCoreMonitorInterval = std::chrono::milliseconds(100),
					const std::chrono::system_clock::duration& gpuMonitorInterval = std::chrono::milliseconds(100));

				/**
				 * Creates a new Monitor.
				 * @param name The name of the Monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				explicit Monitor(std::string name, const std::chrono::system_clock::duration& interval);

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
				 * Determines if the specified variable is defined.
				 * @param timestamp The timestamp.
				 * @param name The name of the variable.
				 * @return Whether the variable is defined.
				 */
				bool hasVariable(const std::chrono::system_clock::time_point& timestamp, const std::string& name) const;

				/**
				 * Gets the specified variable at the specified timestamp.
				 * @param timestamp The timestamp.
				 * @param name The name of the variable.
				 * @return The variable values.
				 */
				std::string getVariable(const std::chrono::system_clock::time_point& timestamp, const std::string& name) const;

				/**
				 * Removes all recorded data and resets the monitor.
				 */
				void reset();

				/**
				 * Determines if the Monitor contains variable values.
				 * @return Whether the Monitor contains variable values.
				 */
				bool hasVariableValues() const;

				/**
				 * Polls all variables and potentially stores their current values.
				 * @param save Whether to save the polled variables.
				 * @return The variables at the current point in time.
				 */
				std::map<std::string, std::string> poll(const bool& save);

				/**
				 * Calculates the difference of a numerical value over the specified time period.
				 * @param variable The variable to inspect.
				 * @param startTimestamp The start timestamp.
				 * @param endTimestamp The end timestamp.
				 * @return The difference.
				 */
				double calculateDifference(const std::string& variable, const std::chrono::system_clock::time_point& startTimestamp, const std::chrono::system_clock::time_point& endTimestamp) const;

				/**
				 * Calculates the difference of a numerical value over the monitoring period.
				 * @param variable The variable to inspect.
				 * @return The difference.
				 */
				double calculateDifference(const std::string& variable) const;

				/**
				 * Calculates the minimum of a numerical value over the specified time period.
				 * @param variable The variable to inspect.
				 * @param startTimestamp The start timestamp.
				 * @param endTimestamp The end timestamp.
				 * @return The minimum.
				 */
				double calculateMinimum(const std::string& variable, const std::chrono::system_clock::time_point& startTimestamp, const std::chrono::system_clock::time_point& endTimestamp) const;

				/**
				 * Calculates the minimum of a numerical value over the monitoring period.
				 * @param variable The variable to inspect.
				 * @return The minimum.
				 */
				double calculateMinimum(const std::string& variable) const;

				/**
				 * Calculates the maximum of a numerical value over the specified time period.
				 * @param variable The variable to inspect.
				 * @param startTimestamp The start timestamp.
				 * @param endTimestamp The end timestamp.
				 * @return The maximum.
				 */
				double calculateMaximum(const std::string& variable, const std::chrono::system_clock::time_point& startTimestamp, const std::chrono::system_clock::time_point& endTimestamp) const;

				/**
				 * Calculates the maximum of a numerical value over the monitoring period.
				 * @param variable The variable to inspect.
				 * @return The maximum.
				 */
				double calculateMaximum(const std::string& variable) const;
			};
		}
	}
}