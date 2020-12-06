#pragma once

#include <chrono>
#include <functional>

namespace EnergyManager {
	namespace Utility {
		/**
		 * Caches a value for a certain amount of time.
		 * @tparam Value The value type.
		 */
		template<typename Value>
		class CachedValue {
			/**
			 * The cached value.
			 */
			Value value_;

			/**
			 * When the value was last updated.
			 */
			std::chrono::system_clock::time_point lastUpdate_ = std::chrono::system_clock::time_point();

			/**
			 * The amount of time to cache the value.
			 */
			std::chrono::system_clock::duration cachePeriod_;

			/**
			 * Mutex to prevent multiple threads from updating and reading the value.
			 */
			std::mutex mutex_;

		public:
			/**
			 * Creates a new cached value.
			 * @param producer Generates a new value when needed.
			 * @param cachePeriod The amount of time to cache the value.
			 */
			CachedValue(const std::chrono::system_clock::duration& cachePeriod = std::chrono::system_clock::duration(0)) : cachePeriod_(cachePeriod) {
			}

			/**
			 * Gets the value that is currently cached.
			 * @param producer Generates the new value when needed.
			 * @return The current value.
			 */
			const Value& getValue(const std::function<Value(const Value& currentValue, const std::chrono::system_clock::duration& timeSinceLastUpdate)>& producer) {
				std::lock_guard<std::mutex> guard(mutex_);

				// Update the value if necessary
				const auto now = std::chrono::system_clock::now();
				const auto timeSinceLastUpdate = now - lastUpdate_;
				if(lastUpdate_ == std::chrono::system_clock::time_point() || timeSinceLastUpdate > cachePeriod_) {
					lastUpdate_ = now;
					value_ = producer(value_, timeSinceLastUpdate);
				}

				return value_;
			}
		};
	}
}