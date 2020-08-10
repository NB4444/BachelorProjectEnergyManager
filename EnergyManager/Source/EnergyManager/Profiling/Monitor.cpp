#include "./Monitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> Monitor::onPoll() {
			return {};
		}

		std::map<std::string, std::string> Monitor::poll() {
			std::map<std::string, std::string> results = onPoll();
			variableValues_[std::chrono::system_clock::now()] = results;

			return results;
		}
	}
}