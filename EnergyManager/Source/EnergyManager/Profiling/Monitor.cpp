#include "./Monitor.hpp"

#include <unistd.h>
#include <utility>

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> Monitor::onPoll() {
			return {};
		}

		Monitor::Monitor(std::string name)
			: name_(std::move(name)) {
		}

		std::string Monitor::getName() const {
			return name_;
		}

		std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>> Monitor::getVariableValues() const {
			return variableValues_;
		}

		std::map<std::string, std::string> Monitor::poll(const bool& save) {
			std::map<std::string, std::string> results = onPoll();

			if(save) {
				variableValues_[std::chrono::system_clock::now()] = results;
			}

			return results;
		}

		void Monitor::run(const std::chrono::seconds& interval) {
			running_ = true;

			while(running_) {
				poll(true);

				usleep(std::chrono::duration_cast<std::chrono::microseconds>(interval).count());
			}
		}

		void Monitor::stop() {
			// Store the final state
			poll(true);

			running_ = false;
		}
	}
}