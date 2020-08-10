#pragma once

#include "Utility/Text.hpp"

#include <map>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

namespace Utility {
	namespace Serialization {
		static std::string serialize(const std::string& value, const std::string& stringEscape = "\\\"") {
			std::string escapedValue;
			std::regex_replace(std::back_inserter(escapedValue), value.begin(), value.end(), std::regex("\""), stringEscape);
			return '"' + escapedValue + '"';
		}

		static std::string serialize(const int& value) {
			return std::to_string(value);
		}

		static std::string serialize(std::vector<std::string> value, const std::string& delimiter = ",", const std::string& stringEscape = "\\\"") {
			// Serialize the items
			std::transform(value.begin(), value.end(), value.begin(), [&](const std::string& item) -> std::string {
				return serialize(item, stringEscape);
			});

			// Join the results
			return Text::join(value, delimiter);
		}

		static std::string serialize(const std::map<std::string, std::string>& value, const std::string& interItemDelimiter = ",", const std::string& intraItemDelimiter = "=>", const std::string& stringEscape = "\\\"") {
			// Serialize the items
			std::vector<std::string> mapItems;
			for(const auto& item : value) {
				mapItems.push_back(serialize(item.first, stringEscape) + intraItemDelimiter + serialize(item.second, stringEscape));
			}

			// Join the results
			return Text::join(mapItems, interItemDelimiter);
		}

		static std::string deserializeToString(const std::string& value, const std::string& stringEscape = "\\\"") {
			std::string unescapedValue;
			std::regex_replace(std::back_inserter(unescapedValue), value.begin(), value.end(), std::regex(stringEscape), "\"");

			// Strip the quotes and return
			return unescapedValue.substr(1, unescapedValue.size() - 2);
		}

		static int deserializeToInt(const std::string& value) {
			return std::stoi(value);
		}

		static std::vector<std::string> deserializeToVectorOfStrings(std::string value, const std::string& delimiter = ",", const std::string& stringEscape = "\\\"") {
			std::vector<std::string> result = Text::split(value, delimiter);
			std::transform(result.begin(), result.end(), result.begin(), [&](const std::string& value) {
				return deserializeToString(value, stringEscape);
			});

			return result;
		}

		static std::map<std::string, std::string> deserializeToMapOfStringsToStrings(const std::string& value, const std::string& interItemDelimiter = ",", const std::string& intraItemDelimiter = "=>", const std::string& stringEscape = "\\\"") {
			std::map<std::string, std::string> result;

			std::vector<std::string> serializedMapItems = deserializeToVectorOfStrings(value, interItemDelimiter);
			for(const auto& serializedMapItem : serializedMapItems) {
				auto mapItem = deserializeToVectorOfStrings(serializedMapItem, intraItemDelimiter, stringEscape);
				result[deserializeToString(mapItem[0], stringEscape)] = deserializeToString(mapItem[1], stringEscape);
			}

			return result;
		}
	}
}