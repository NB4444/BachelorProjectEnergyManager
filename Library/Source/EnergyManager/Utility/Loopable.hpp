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
			bool isLooping_ = true;

			/**
			 * The mutex that protects the looping state.
			 */
			std::mutex isLoopingMutex_;

			/**
			 * The condition that determines whether to keep looping.
			 */
			std::condition_variable loopCondition_;

		protected:
			void onRun() final;

			/**
			 * Executes when the object is looped.
			 */
			virtual void onLoop();

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