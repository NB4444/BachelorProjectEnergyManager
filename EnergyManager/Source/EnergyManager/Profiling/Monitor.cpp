#include "./Monitor.hpp"

#include <unistd.h>
#include <utility>

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> Monitor::onPoll() {
			return {};
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
			std::map<std::string, std::string> results = onPoll();

			lastPollTimestamp_ = std::chrono::system_clock::now();

			if(save) {
				auto now = std::chrono::system_clock::now();

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
				poll(true);

				usleep(std::chrono::duration_cast<std::chrono::microseconds>(interval).count());
			}
		}

		void Monitor::stop() {
			// Store the final state
			poll(true);

			isRunning_ = false;
		}
	}
}