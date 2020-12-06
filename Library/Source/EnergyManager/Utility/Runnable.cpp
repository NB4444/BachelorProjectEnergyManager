#include "./Runnable.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Utility {
		void Runnable::beforeRun() {
		}

		void Runnable::onRun() {
		}

		void Runnable::afterRun() {
		}

		void Runnable::sleep(const std::chrono::system_clock::duration& duration) {
			usleep(std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
		}

		Runnable::Runnable() : isRunning_(false) {
		}

		Runnable::Runnable(const Runnable& runnable) {
			if(isRunning()) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Running objects cannot be copied");
			}
		}

		bool Runnable::isRunning() const {
			return isRunning_;
		}

		std::chrono::system_clock::time_point Runnable::getStartTimestamp() const {
			return startTimestamp_;
		}

		std::chrono::system_clock::duration Runnable::getRuntime() const {
			return std::chrono::system_clock::now() - getStartTimestamp();
		}

		void Runnable::run(const bool& asynchronous) {
			logTrace("Running runnable" + std::string(asynchronous ? " asynchronously" : "") + "...");

			auto operation = [&] {
				//{
				//	// Set up the running state and lock any waiting threads
				//	std::unique_lock<std::mutex> lock(synchronizationMutex_);
				isRunning_ = true;

				logTrace("Preparing run...");
				beforeRun();

				// Run the operation
				logTrace("Executing workload...");
				onRun();

				logTrace("Finalizing run...");
				afterRun();

				logTrace("Finished workload");

				// Release any waiting threads
				isRunning_ = false;
				//}
				//
				//logTrace("Notifying waiting threads...");
				//synchronizationCondition_.notify_all();
			};

			// Run the operation
			startTimestamp_ = std::chrono::system_clock::now();
			if(asynchronous) {
				runThread_ = std::thread(operation);
				Logging::registerThread(runThread_);
			} else {
				operation();
			}
		}

		void Runnable::synchronize() {
			logTrace("Synchronizing with run thread...");

			//std::unique_lock<std::mutex> lock(synchronizationMutex_);

			// Join the thread to wait for it to stop, if there is a thread
			if(runThread_.joinable()) {
				logTrace("Waiting for worker thread to finish...");

				// Wait for the worker thread to finish running
				while(isRunning_) {
					sleep(std::chrono::milliseconds(1));
				}
				//synchronizationCondition_.wait(lock, [&] {
				//	return !isRunning_;
				//});

				logTrace("Joining thread...");

				runThread_.join();
			}

			logTrace("Synchronized with run thread");
		}
	}
}