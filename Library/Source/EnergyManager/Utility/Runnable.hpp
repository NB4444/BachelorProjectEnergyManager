#pragma once

#include "EnergyManager/Utility/Logging/Loggable.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
//#include <mutex>
#include <thread>
#include <unistd.h>

namespace EnergyManager {
	namespace Utility {
		/**
		 * An object that can be executed.
		 */
		class Runnable : protected Logging::Loggable {
			/**
			 * The start timestamp.
			 */
			std::chrono::system_clock::time_point startTimestamp_;

			/**
			 * The thread that is currently running the object.
			 */
			std::thread runThread_;

			/**
			 * Whether the object is running.
			 */
			std::atomic<bool> isRunning_;

			///**
			// * The mutex to protect the conditional variable.
			// */
			//std::mutex synchronizationMutex_;
			//
			///**
			// * The conditional variable to share the running state between threads and allow synchronization.
			// */
			//std::condition_variable synchronizationCondition_;

		protected:
			/**
			 * Executes just before the object starts running.
			 */
			virtual void beforeRun();

			/**
			 * Executes when the object is run.
			 */
			virtual void onRun();

			/**
			 * Executes after the object has finished running.
			 */
			virtual void afterRun();

		public:
			/**
			 * Suspends the current thread for the specified amount of time.
			 * @param duration The time to sleep.
			 */
			static void sleep(const std::chrono::system_clock::duration& duration);

			/**
			 * Creates a new Runnable.
			 */
			Runnable();

			/**
			 * Copies the object.
			 * @param runnable The object to copy.
			 */
			Runnable(const Runnable& runnable);

			/**
			 * Gets whether the object is running.
			 * @return Whether the object is running.
			 */
			bool isRunning() const;

			/**
			 * Gets the start timestamp.
			 * @return The start timestamp.
			 */
			std::chrono::system_clock::time_point getStartTimestamp() const;

			/**
			 * Gets the total runtime.
			 * @return The runtime.
			 */
			std::chrono::system_clock::duration getRuntime() const;

			/**
			 * Runs the object.
			 * @param asynchronous Whether to use a new thread to run the object.
			 */
			void run(const bool& asynchronous = false);

			/**
			 * Waits for the object to finish running.
			 * Useful for asynchronous operations.
			 */
			void synchronize();
		};
	}
}