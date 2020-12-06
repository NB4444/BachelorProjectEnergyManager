#pragma once

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <vector>

namespace EnergyManager {
	namespace Utility {
		namespace Collections {
			/**
			 * Splits a vector of items into chunks of a maximum size.
			 * @tparam Type The type of the items.
			 * @param values The items.
			 * @param maximumChunkSize The maximum size of one chunk.
			 * @return The chunks.
			 */
			template<typename Type>
			static std::vector<std::vector<Type>> splitInChunks(const std::vector<Type>& values, const unsigned int& maximumChunkSize) {
				std::vector<std::vector<Type>> chunks = {};

				for(const auto& value : values) {
					// Create new chunk if the last one was full
					if(chunks.empty() || chunks.back().size() >= maximumChunkSize) {
						chunks.push_back({});
					}

					// Append the current item to the chunk
					chunks.back().push_back(value);
				}

				return chunks;
			}

			template<typename Key, typename Value>
			static typename std::map<Key, Value>::value_type& getMapItemByIndex(std::map<Key, Value>& map, const unsigned int& index) {
				unsigned int currentIndex = 0;
				for(auto& item : map) {
					if(currentIndex == index) {
						return item;
					}
					++currentIndex;
				}

				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Index out of range");
			}
		}
	}
}