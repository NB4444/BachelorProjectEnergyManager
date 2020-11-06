#include "./Monitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			void Monitor::onLoop() {
				poll(true);
			}

			std::map<std::string, std::string> Monitor::onPoll() {
				return std::map<std::string, std::string>();
			}

			void Monitor::onReset() {
			}

			Monitor::Monitor(std::string name, const std::chrono::system_clock::duration& interval) : Loopable(interval), name_(std::move(name)) {
			}

			std::string Monitor::getName() const {
				return name_;
			}

			std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> Monitor::getVariableValues() const {
				return variableValues_;
			}

			void Monitor::reset() {
				variableValues_.clear();

				onReset();
			}

			bool Monitor::hasVariableValues() const {
				return !variableValues_.empty();
			}

			std::map<std::string, std::string> Monitor::poll(const bool& save) {
				auto now = std::chrono::system_clock::now();

				std::map<std::string, std::string> results = onPoll();

				if(save) {
					variableValues_[now] = results;
					variableValues_[now]["runtime"] = std::to_string(getRuntime().count());
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