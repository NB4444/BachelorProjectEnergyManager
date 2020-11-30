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
				// Lock synchronization
				std::unique_lock<std::mutex> lock(synchronizationMutex_);

				logTrace("Preparing run...");
				beforeRun();

				// Set up the running state and lock any waiting threads
				isRunning_ = true;
				startTimestamp_ = std::chrono::system_clock::now();

				// Run the operation
				logTrace("Executing workload...");
				onRun();

				// Release any waiting threads
				isRunning_ = false;

				logTrace("Finalizing run...");
				afterRun();

				// Unlock the waiting synchronization
				lock.unlock();
				synchronizationCondition_.notify_one();
			};
			if(asynchronous) {
				runThread_ = std::thread(operation);
				Logging::registerThread(runThread_);
			} else {
				operation();
			}
		}

		void Runnable::synchronize() {
			logTrace("Synchronizing with run thread...");

			// Wait for the worker thread to finish running
			std::unique_lock<std::mutex> lock(synchronizationMutex_);
			synchronizationCondition_.wait(lock, [&] {
				return !isRunning_;
			});

			// Join the thread to wait for it to stop, if there is a thread
			if(runThread_.joinable()) {
				runThread_.join();
			}
		}
	}
}