#pragma once

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace Utility {
	namespace Text {
		static std::string join(const std::vector<std::string>& value, const std::string& delimiter) {
			return std::accumulate(value.begin(), value.end(), std::string(), [&](const std::string& left, const std::string& right) -> std::string {
				return left + (left.length() > 0 ? delimiter : "") + right;
			});
		}

		static std::vector<std::string> split(std::string value, const std::string& delimiter) {
			std::vector<std::string> result;

			size_t index = 0u;
			while((index = value.find(delimiter)) != std::string::npos) {
				std::string segment = value.substr(0, index);
				result.push_back(segment);
				value.erase(0u, index + delimiter.length());
			}
			result.push_back(value);

			return result;
		}

		static std::string trim(std::string value) {
			// Trim prefix whitespace
			value.erase(
				value.begin(),
				std::find_if(value.begin(), value.end(), [](const int& character) {
					return !std::isspace(character);
				}));

			// Trim postfix whitespace
			value.erase(
				std::find_if(value.rbegin(), value.rend(), [](const int& character) {
					return !std::isspace(character);
				}).base(),
				value.end());

			return value;
		}
	}
}