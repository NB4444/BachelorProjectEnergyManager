#include "./Runnable.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

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
			auto operation = [&] {
				beforeRun();

				// Set up the running state and lock any waiting threads
				std::unique_lock<std::mutex> lock(synchronizationMutex_);
				isRunning_ = true;
				startTimestamp_ = std::chrono::system_clock::now();

				// Run the operation
				onRun();

				// Release any waiting threads
				isRunning_ = false;
				lock.unlock();
				synchronizationCondition_.notify_one();

				afterRun();
			};
			if(asynchronous) {
				runThread_ = std::thread(operation);
			} else {
				operation();
			}
		}

		void Runnable::synchronize() {
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