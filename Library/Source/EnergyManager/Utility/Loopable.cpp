#include "./Loopable.hpp"

#include "EnergyManager/Configuration.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Utility {
		void Loopable::onRun() {
			logTrace("Starting loopable...");

			std::chrono::system_clock::time_point nextRun {};
			bool ranOnce = false;

			while(true) {
				{
					std::unique_lock<std::mutex> lock(isLoopingMutex_);
					if(!isLooping_ && ranOnce) {
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
						if(Configuration::warningWhenLoopIntervalExceeded) {
							logWarning("Can't keep up, exceeded loop interval by %s", Utility::Text::formatDuration(difference).c_str());
						}

						// Make the next run earlier by the amount of time we were delayed
						nextRun = now;
						if(difference < interval_) {
							nextRun += interval_ - difference;
						} else if(Configuration::warningWhenSkippingLoopIterations) {
							logWarning("Skipped a loop iteration");
						}
					} else {
						const auto difference = nextRun - now;
						logTrace("Waiting for next loop to start in %s...", Utility::Text::formatDuration(difference).c_str());

						loopCondition_.wait_for(lock, difference);
						nextRun = std::chrono::system_clock::now() + interval_;
					}
				}

				loop();
			}

			// Reset the looping state
			isLooping_ = true;
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
			logTrace("Looping loopable...");

			auto now = std::chrono::system_clock::now();

			lastLoopTimestamp_ = now;

			// Do the loop
			onLoop();

			// Flush the output from the loop
			Logging::flush();
		}

		void Loopable::stop(const bool& synchronize) {
			logTrace("Stopping loopable...");

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