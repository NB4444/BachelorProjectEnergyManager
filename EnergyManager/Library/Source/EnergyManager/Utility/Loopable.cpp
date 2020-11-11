#include "./Loopable.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

namespace EnergyManager {
	namespace Utility {
		void Loopable::onRun() {
			isLooping_ = true;

			std::chrono::system_clock::time_point nextRun {};
			bool ranOnce = false;

			while(true) {
				{
					std::unique_lock<std::mutex> lock(isLoopingMutex_);
					if(!isLooping_) {
						break;
					}

					// Initialize the next run tracker
					const auto now = std::chrono::system_clock::now();
					if(!ranOnce) {
						ranOnce = true;
						nextRun = now;
					}

					// Check if we have exceeded the interval
					if(now > nextRun) {
						const auto difference = now - nextRun;
						Logging::logWarning("Can't keep up, exceeded loop interval by %s", Utility::Text::formatDuration(difference).c_str());

						// Make the next run earlier by the amount of time we were delayed
						nextRun = now;
						if(difference < interval_) {
							nextRun += interval_ - difference;
						} else {
							Logging::logWarning("Skipped a monitor frame");
						}
					} else {
						// If not, wait until the end of the interval
						loopCondition_.wait_for(lock, nextRun - now);
						nextRun = std::chrono::system_clock::now() + interval_;
					}
				}

				loop();
			}
		}

		void Loopable::onLoop() {
		}

		Loopable::Loopable(const std::chrono::system_clock::duration& interval) : interval_(interval) {
		}

		Loopable::~Loopable() {
			stop(true);
		}

		std::chrono::system_clock::time_point Loopable::getLastLoopTimestamp() const {
			return lastLoopTimestamp_;
		}

		std::chrono::system_clock::duration Loopable::getTimeSinceLastLoop() const {
			return std::chrono::system_clock::now() - getLastLoopTimestamp();
		}

		void Loopable::loop() {
			auto now = std::chrono::system_clock::now();

			lastLoopTimestamp_ = now;

			onLoop();
		}

		void Loopable::stop(const bool& synchronize) {
			{
				std::unique_lock<std::mutex> lock(isLoopingMutex_);

				// Break the loop
				isLooping_ = false;
			}
			loopCondition_.notify_one();

			if(synchronize) {
				this->synchronize();
			}
		}
	}
}