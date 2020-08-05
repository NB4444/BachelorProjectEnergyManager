#pragma once

#include <map>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

namespace Utility::Serialization {
	static std::string serialize(const std::string& value) {
		return value;
	}

	static std::string serialize(const int& value) {
		return std::to_string(value);
	}

	static std::string serialize(const std::vector<std::string>& value, const std::string& delimiter = ",") {
		return std::accumulate(value.begin(), value.end(), std::string(), [&](const std::string& left, const std::string& right) -> std::string {
			return left + (left.length() > 0 ? delimiter : "") + right;
		});
	}

	static std::string serialize(const std::map<std::string, std::string>& value, const std::string& interItemDelimiter = ",", const std::string& intraItemDelimiter = "=>") {
		std::vector<std::string> mapItems;

		for(const auto& item : value) {
			mapItems.push_back(item.first + intraItemDelimiter + item.second);
		}

		return serialize(mapItems, interItemDelimiter);
	}

	static std::string deserializeToString(const std::string& value) {
		return value;
	}

	static int deserializeToInt(const std::string& value) {
		return std::stoi(value);
	}

	static std::vector<std::string> deserializeToVectorOfStrings(std::string value, const std::string& delimiter = ",") {
		std::vector<std::string> result;

		size_t index = 0u;
		while((index = value.find(delimiter)) != std::string::npos) {
			std::string segment = value.substr(0, index);
			result.push_back(segment);
			value.erase(0u, index + delimiter.length());
		}

		return result;
	}

	static std::map<std::string, std::string> deserializeToMapOfStringsToStrings(const std::string& value, const std::string& interItemDelimiter = ",", const std::string& intraItemDelimiter = "=>") {
		std::map<std::string, std::string> result;

		std::vector<std::string> serializedMapItems = deserializeToVectorOfStrings(value, interItemDelimiter);
		for(const auto& serializedMapItem : serializedMapItems) {
			auto mapItem = deserializeToVectorOfStrings(serializedMapItem, intraItemDelimiter);
			result[mapItem[0]] = mapItem[1];
		}

		return result;
	}
}