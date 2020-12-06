#include "./Monitor.hpp"

#include "EnergyManager/Monitoring/Monitors/CPUCoreMonitor.hpp"
#include "EnergyManager/Monitoring/Monitors/CPUMonitor.hpp"
#include "EnergyManager/Monitoring/Monitors/GPUMonitor.hpp"
#include "EnergyManager/Monitoring/Monitors/NodeMonitor.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::vector<std::string> Monitor::generateHeaders() const {
				auto headers = Runnable::generateHeaders();
				headers.push_back("Monitor " + getName());

				return headers;
			}

			void Monitor::onLoop() {
				poll(true);
			}

			std::map<std::string, std::string> Monitor::onPoll() {
				return std::map<std::string, std::string>();
			}

			void Monitor::onReset() {
			}

			void Monitor::setVariable(const std::chrono::system_clock::time_point& timestamp, const std::string& name, const std::string& value) {
				// Insert the map if it does not exist
				if(variableValues_.find(timestamp) == variableValues_.end()) {
					variableValues_[timestamp] = {};
				}

				// Set the variable
				variableValues_[timestamp][name] = value;
			}

			std::vector<std::shared_ptr<Monitor>> Monitor::getMonitorsForAllDevices(
				const std::chrono::system_clock::duration& applicationMonitorInterval,
				const std::chrono::system_clock::duration& nodeMonitorInterval,
				const std::chrono::system_clock::duration& cpuMonitorInterval,
				const std::chrono::system_clock::duration& cpuCoreMonitorInterval,
				const std::chrono::system_clock::duration& gpuMonitorInterval) {
				std::vector<std::shared_ptr<Monitor>> monitors = { std::make_shared<NodeMonitor>(Hardware::Node::getNode(), nodeMonitorInterval) };
				for(const auto& cpu : Hardware::CPU::getCPUs()) {
					monitors.push_back(std::make_shared<CPUMonitor>(cpu, cpuMonitorInterval));

					for(const auto& core : cpu->getCores()) {
						monitors.push_back(std::make_shared<CPUCoreMonitor>(core, cpuMonitorInterval));
					}
				}
				for(const auto& gpu : Hardware::GPU::getGPUs()) {
					monitors.push_back(std::make_shared<GPUMonitor>(gpu, gpuMonitorInterval));
				}

				return monitors;
			}

			Monitor::Monitor(std::string name, const std::chrono::system_clock::duration& interval) : Loopable(interval), name_(std::move(name)) {
			}

			std::string Monitor::getName() const {
				return name_;
			}

			std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> Monitor::getVariableValues() const {
				return variableValues_;
			}

			bool Monitor::hasVariable(const std::chrono::system_clock::time_point& timestamp, const std::string& name) const {
				if(variableValues_.find(timestamp) == variableValues_.end()) {
					return false;
				} else {
					const auto& values = variableValues_.at(timestamp);

					return values.find(name) != values.end();
				}
			}

			std::string Monitor::getVariable(const std::chrono::system_clock::time_point& timestamp, const std::string& name) const {
				return variableValues_.at(timestamp).at(name);
			}

			void Monitor::reset() {
				variableValues_.clear();

				onReset();
			}

			bool Monitor::hasVariableValues() const {
				return !variableValues_.empty();
			}

			std::map<std::string, std::string> Monitor::poll(const bool& save) {
				logTrace("Polling monitor...");

				std::map<std::string, std::string> results = onPoll();

				if(save) {
					variableValues_[getLastLoopTimestamp()] = results;
					variableValues_[getLastLoopTimestamp()]["runtime"] = Utility::Text::toString(getRuntime());
				}

				return results;
			}

			double Monitor::calculateDifference(const std::string& variable, const std::chrono::system_clock::time_point& startTimestamp, const std::chrono::system_clock::time_point& endTimestamp)
				const {
				const auto data = getVariableValues();

				// Find the first and last data point
				auto firstDataPointIterator = data.lower_bound(startTimestamp);
				if(firstDataPointIterator == data.end()) {
					firstDataPointIterator = data.begin();
				}
				auto lastDataPointIterator = data.upper_bound(endTimestamp);
				if(lastDataPointIterator == data.end()) {
					lastDataPointIterator = --data.end();
				}

				// Calculate the difference
				return std::stod(lastDataPointIterator->second.at(variable)) - std::stod(firstDataPointIterator->second.at(variable));
			}

			double Monitor::calculateDifference(const std::string& variable) const {
				return calculateDifference(variable, getStartTimestamp(), std::chrono::system_clock::now());
			}

			double
				Monitor::calculateMinimum(const std::string& variable, const std::chrono::system_clock::time_point& startTimestamp, const std::chrono::system_clock::time_point& endTimestamp) const {
				const auto data = getVariableValues();

				// Find the first and last data point
				auto firstDataPointIterator = data.lower_bound(startTimestamp);
				if(firstDataPointIterator == data.end()) {
					firstDataPointIterator = data.begin();
				}
				auto lastDataPointIterator = data.upper_bound(endTimestamp);
				if(lastDataPointIterator == data.end()) {
					lastDataPointIterator = --data.end();
				}

				// Find the minimum value
				bool found = false;
				double minimum;
				for(auto iterator = firstDataPointIterator; iterator != lastDataPointIterator; ++iterator) {
					const auto value = std::stod(iterator->second.at(variable));

					if(!found || value < minimum) {
						minimum = value;
						found = true;
					}
				}

				return minimum;
			}

			double Monitor::calculateMinimum(const std::string& variable) const {
				return calculateMinimum(variable, getStartTimestamp(), std::chrono::system_clock::now());
			}

			double
				Monitor::calculateMaximum(const std::string& variable, const std::chrono::system_clock::time_point& startTimestamp, const std::chrono::system_clock::time_point& endTimestamp) const {
				const auto data = getVariableValues();

				// Find the first and last data point
				auto firstDataPointIterator = data.lower_bound(startTimestamp);
				if(firstDataPointIterator == data.end()) {
					firstDataPointIterator = data.begin();
				}
				auto lastDataPointIterator = data.upper_bound(endTimestamp);
				if(lastDataPointIterator == data.end()) {
					lastDataPointIterator = --data.end();
				}

				// Find the maximum value
				bool found = false;
				double maximum;
				for(auto iterator = firstDataPointIterator; iterator != lastDataPointIterator; ++iterator) {
					const auto value = std::stod(iterator->second.at(variable));

					if(!found || value > maximum) {
						maximum = value;
						found = true;
					}
				}

				return maximum;
			}

			double Monitor::calculateMaximum(const std::string& variable) const {
				return calculateMaximum(variable, getStartTimestamp(), std::chrono::system_clock::now());
			}
		}
	}
}