#include "./Monitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <unistd.h>
#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		std::vector<Monitor::Parser> Monitor::parsers_ = {};

		void Monitor::addParser(const Monitor::Parser& parser) {
			parsers_.push_back(parser);
		}

		std::map<std::string, std::string> Monitor::onPoll() {
			return {};
		}

		std::shared_ptr<Monitor> Monitor::parse(const std::string& name, const std::map<std::string, std::string>& parameters) {
			for(const auto& parser : parsers_) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(return parser(name, parameters));
			}

			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not parse Monitor");
		}

		Monitor::Monitor(std::string name) : name_(std::move(name)) {
		}

		std::string Monitor::getName() const {
			return name_;
		}

		std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> Monitor::getVariableValues() const {
			return variableValues_;
		}

		std::chrono::system_clock::time_point Monitor::getStartTimestamp() const {
			return startTimestamp_;
		}

		std::chrono::system_clock::duration Monitor::getRuntime() const {
			return std::chrono::system_clock::now() - getStartTimestamp();
		}

		std::chrono::system_clock::time_point Monitor::getLastPollTimestamp() const {
			return lastPollTimestamp_;
		}

		std::chrono::system_clock::duration Monitor::getTimeSinceLastPoll() const {
			return std::chrono::system_clock::now() - getLastPollTimestamp();
		}

		bool Monitor::isRunning() const {
			return isRunning_;
		}

		std::map<std::string, std::string> Monitor::poll(const bool& save) {
			auto now = std::chrono::system_clock::now();

			std::map<std::string, std::string> results = onPoll();

			lastPollTimestamp_ = now;

			if(save) {
				variableValues_[now] = results;
				variableValues_[now]["runtime"] = std::to_string(getRuntime().count());
			}

			return results;
		}

		void Monitor::run(const std::chrono::system_clock::duration& interval) {
			isRunning_ = true;
			startTimestamp_ = std::chrono::system_clock::now();
			lastPollTimestamp_ = startTimestamp_;

			while(isRunning_) {
				if((std::chrono::system_clock::now() - lastPollTimestamp_) >= interval) {
					poll(true);
				} else {
					usleep(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::milliseconds(10)).count());
				}
			}
		}

		void Monitor::stop() {
			// Store the final state
			poll(true);

			isRunning_ = false;
		}
	}
}