#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <time.h>
#include <vector>

namespace EnergyManager {
	namespace Utility {
		namespace Text {
			template<typename Type>
			static std::string toString(const Type& value) {
				return static_cast<std::string>(value);
			}

			template<>
			std::string toString<unsigned int>(const unsigned int& value) {
				return std::to_string(value);
			}

			template<>
			std::string toString<unsigned long>(const unsigned long& value) {
				return std::to_string(value);
			}

			template<>
			std::string toString<int>(const int& value) {
				return std::to_string(value);
			}

			template<>
			std::string toString<long>(const long& value) {
				return std::to_string(value);
			}

			static std::string trim(std::string value) {
				// Trim prefix whitespace
				value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](const int& character) {
								return !std::isspace(character);
							}));

				// Trim postfix whitespace
				value.erase(
					std::find_if(
						value.rbegin(),
						value.rend(),
						[](const int& character) {
							return !std::isspace(character);
						})
						.base(),
					value.end());

				return value;
			}

			static std::string mergeWhitespace(const std::string& value) {
				return std::regex_replace(value, std::regex("\\s+"), " ");
			}

			template<typename Type>
			static std::string join(const std::vector<Type>& value, const std::string& delimiter) {
				std::string result;

				for(size_t index = 0u; index < value.size(); ++index) {
					if(index != 0u) {
						result += delimiter;
					}

					result += toString(value[index]);
				}

				return result;
			}

			template<typename Type>
			static std::string join(const std::set<Type>& value, const std::string& delimiter) {
				return join(std::vector<Type>(value.begin(), value.end()), delimiter);
			}

			static std::vector<std::string> splitToVector(std::string value, const std::string& pairDelimiter, const bool& trim = false) {
				std::vector<std::string> result;

				size_t index = 0u;
				while((index = value.find(pairDelimiter)) != std::string::npos) {
					// Get the current segment and add it
					std::string segment = value.substr(0, index);
					if(trim) {
						segment = Text::trim(segment);
					}
					result.push_back(segment);

					// Remove the segment from the remainder
					value.erase(0u, index + pairDelimiter.length());
				}
				result.push_back(value);

				return result;
			}

			static std::map<std::string, std::string> splitToMap(const std::string& value, const std::string& pairDelimiter, const std::string& itemDelimiter, const bool& trim = false) {
				// First extract the items
				std::vector<std::string> items = splitToVector(value, itemDelimiter, trim);

				// Then, for each item, extract the key and value and add them to the result
				std::map<std::string, std::string> result;
				for(const auto& item : items) {
					std::vector<std::string> keyValue = splitToVector(item, pairDelimiter, trim);
					result[keyValue[0]] = keyValue[1];
				}

				return result;
			}

			static std::string formatTimestamp(const std::time_t& timestamp, const std::string& format) {
				// Get the current date and time
				auto localTime = *std::localtime(&timestamp);
				std::ostringstream outputStringStream;
				outputStringStream << std::put_time(&localTime, format.c_str());

				return outputStringStream.str();
			}

			static std::string formatTimestamp(const std::chrono::system_clock::time_point& timestamp, const std::string& format = "%Y-%m-%d %H:%M:%S") {
				return formatTimestamp(std::chrono::system_clock::to_time_t(timestamp), format);
			}
		}
	}
}