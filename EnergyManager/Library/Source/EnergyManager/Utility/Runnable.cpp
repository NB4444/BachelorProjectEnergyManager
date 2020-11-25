#include "./Runnable.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Utility {
		std::map<std::thread::id, unsigned int> Runnable::threadIDs_ = {};

		std::vector<std::string> Runnable::generateHeaders() const {
			return { "Thread " + Text::toString(getThreadID()) };
		}

		void Runnable::beforeRun() {
		}

		void Runnable::onRun() {
		}

		void Runnable::afterRun() {
		}

		unsigned int Runnable::getCurrentThreadID() {
			return threadIDs_.at(std::this_thread::get_id());
		}

		Runnable::Runnable(const Runnable& runnable) {
			if(isRunning()) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Running objects cannot be copied");
			}
		}

		unsigned int Runnable::getThreadID() const {
			return threadIDs_[runThread_.get_id()];
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
			logDebug("Running runnable" + std::string(asynchronous ? " asynchronously" : "") + "...");

			auto operation = [&] {
				logDebug("Preparing run...");
				beforeRun();

				// Set up the running state and lock any waiting threads
				std::unique_lock<std::mutex> lock(synchronizationMutex_);
				isRunning_ = true;
				startTimestamp_ = std::chrono::system_clock::now();

				// Run the operation
				logDebug("Executing workload...");
				onRun();

				// Release any waiting threads
				isRunning_ = false;
				lock.unlock();
				synchronizationCondition_.notify_one();

				logDebug("Finalizing run...");
				afterRun();
			};
			if(asynchronous) {
				runThread_ = std::thread(operation);
				threadIDs_[runThread_.get_id()] = nextThreadID_++;
			} else {
				operation();
			}
		}

		void Runnable::synchronize() {
			logDebug("Synchronizing with run thread...");

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