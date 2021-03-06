#pragma once

#include "EnergyManager/Utility/Logging/Loggable.hpp"
#include "EnergyManager/Utility/Runnable.hpp"

namespace EnergyManager {
	namespace Utility {
		/**
		 * An object that can be executed in a loop with a specified interval.
		 */
		class Loopable : public Runnable {
			/**
			 * The interval between loops.
			 */
			std::chrono::system_clock::duration interval_;

			/**
			 * The last loop timestamp.
			 */
			std::chrono::system_clock::time_point lastLoopTimestamp_;

			/**
			 * Whether the object is currently still looping.
			 */
			std::atomic<bool> isLooping_;

			///**
			// * The mutex that protects the looping state.
			// */
			//std::mutex isLoopingMutex_;
			//
			///**
			// * The condition that determines whether to keep looping.
			// */
			//std::condition_variable loopCondition_;

		protected:
			void beforeRun() final;

			void onRun() final;

			void afterRun() final;

			/**
			 * Executes before the object starts looping.
			 */
			virtual void beforeLoopStart();

			/**
			 * Executes before the object starts looping.
			 */
			virtual void afterLoopEnd();

			/**
			 * Executes before the object does a loop looping.
			 */
			virtual void beforeLoop();

			/**
			 * Executes when the object is looped.
			 */
			virtual void onLoop();

			/**
			 * Executes after the object does a loop looping.
			 */
			virtual void afterLoop();

		public:
			/**
			 * Creates a new Loopable.
			 * @param interval The interval between loops.
			 */
			explicit Loopable(const std::chrono::system_clock::duration& interval);

			/**
			 * Stops the thread.
			 */
			~Loopable();

			/**
			 * Gets the interval at which the Loopable loops.
			 * @return The interval.
			 */
			std::chrono::system_clock::duration getInterval() const;

			/**
			 * Gets the timestamp of the last loop operation.
			 * @return The last loop timestamp.
			 */
			std::chrono::system_clock::time_point getLastLoopTimestamp() const;

			/**
			 * Gets the time since the last loop operation.
			 * @return The time since the last loop operation.
			 */
			std::chrono::system_clock::duration getTimeSinceLastLoop() const;

			/**
			 * Executes one loop.
			 */
			void loop();

			/**
			 * Stops the object from looping.
			 * @param synchronize Whether to wait for any underlying threads to exit.
			 */
			void stop(const bool& synchronize = false);
		};
	}
}