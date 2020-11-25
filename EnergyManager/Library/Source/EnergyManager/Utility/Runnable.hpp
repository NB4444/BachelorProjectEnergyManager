#pragma once

#include "EnergyManager/Utility/Logging/Loggable.hpp"

#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>

namespace EnergyManager {
	namespace Utility {
		/**
		 * An object that can be executed.
		 */
		class Runnable : protected Logging::Loggable {
			/**
			 * Human-understandable thread IDs.
			 */
			static std::map<std::thread::id, unsigned int> threadIDs_;

			/**
			 * Keep track of the next thread ID.
			 */
			unsigned int nextThreadID_ = 0;

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
			bool isRunning_ = false;

			/**
			 * The mutex to protect the conditional variable.
			 */
			std::mutex synchronizationMutex_;

			/**
			 * The conditional variable to share the running state between threads and allow synchronization.
			 */
			std::condition_variable synchronizationCondition_;

		protected:
			std::vector<std::string> generateHeaders() const override;

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
			 * Gets the ID of the current thread.
			 * @return The thread ID.
			 */
			static unsigned int getCurrentThreadID();

			/**
			 * Creates a new Runnable.
			 */
			Runnable() = default;

			/**
			 * Copies the object.
			 * @param runnable The object to copy.
			 */
			Runnable(const Runnable& runnable);

			/**
			 * Gets the ID of the associated thread.
			 * @return The thread ID.
			 */
			unsigned int getThreadID() const;

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