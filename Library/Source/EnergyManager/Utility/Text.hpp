#pragma once

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace EnergyManager {
	namespace Utility {
		namespace Text {
			/**
             * Converts the value to a string.
             * @tparam Type The type of the value.
             * @param value The value.
             * @return A string representation.
             */
			template<typename Type>
			static std::string toString(const Type& value) {
				return static_cast<std::string>(value);
			}

			/**
             * Converts a bool to a string.
             * @param value The bool.
             * @return A string representation.
             */
			template<>
			std::string toString<bool>(const bool& value) {
				return std::to_string(value);
			}

			/**
             * Converts an unsigned int to a string.
             * @param value The unsigned int.
             * @return A string representation.
             */
			template<>
			std::string toString<unsigned int>(const unsigned int& value) {
				return std::to_string(value);
			}

			/**
             * Converts an unsigned long to a string.
             * @param value The unsigned long.
             * @return A string representation.
             */
			template<>
			std::string toString<unsigned long>(const unsigned long& value) {
				return std::to_string(value);
			}

			/**
             * Converts an int to a string.
             * @param value The int.
             * @return A string representation.
             */
			template<>
			std::string toString<int>(const int& value) {
				return std::to_string(value);
			}

			/**
             * Converts a long to a string.
             * @param value The long.
             * @return A string representation.
             */
			template<>
			std::string toString<long>(const long& value) {
				return std::to_string(value);
			}

			/**
             * Converts a double to a string.
             * @param value The double.
             * @return A string representation.
             */
			template<>
			std::string toString<double>(const double& value) {
				return std::to_string(value);
			}

			/**
             * Converts a duration to a string.
             * @param value The duration.
             * @return A string representation.
             */
			template<>
			std::string toString<std::chrono::system_clock::duration>(const std::chrono::system_clock::duration& value) {
				return std::to_string(value.count());
			}

			/**
             * Converts a time point to a string.
             * @param value The time point.
             * @return A string representation.
             */
			template<>
			std::string toString<std::chrono::system_clock::time_point>(const std::chrono::system_clock::time_point& value) {
				return std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count());
			}

			/**
			 * Converts a string to a time point.
			 * @param value The string.
			 * @return A time point representation.
			 */
			static std::chrono::system_clock::time_point timestampFromString(const std::string& value) {
				return std::chrono::system_clock::time_point(std::chrono::nanoseconds(std::stoull(value)));
			}

			/**
             * Trims the whitespace from a string.
             * @param value The string to trim.
             * @return The trimmed string.
             */
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

			/**
             * Merge multiple whitespace characters into one character in the specified string.
             * @param value The string to process.
             * @return The string with whitespace merged.
             */
			static std::string mergeWhitespace(const std::string& value) {
				return std::regex_replace(value, std::regex("\\s+"), " ");
			}

			/**
             * Joins the values in a vector.
             * @tparam Type The type of the vector values.
             * @param value The vector to join.
             * @param delimiter The delimiter string to put between values.
             * @return A string representation.
             */
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

			/**
             * Joins the values in a set.
             * @tparam Type The type of the set values.
             * @param value The set to join.
             * @param delimiter The delimiter string to put between values.
             * @return A string representation.
             */
			template<typename Type>
			static std::string join(const std::set<Type>& value, const std::string& delimiter) {
				return join(std::vector<Type>(value.begin(), value.end()), delimiter);
			}

			/**
             * Joins the values in a map.
             * @tparam KeyType The type of the map keys.
             * @tparam ValueType The type of the map values.
             * @param value The map to join.
             * @param itemDelimiter The delimiter between items.
             * @param keyValueDelimiter The delimiter between the key and value.
             * @return A string representation.
             */
			template<typename KeyType, typename ValueType>
			static std::string join(const std::map<KeyType, ValueType>& value, const std::string& itemDelimiter, const std::string& keyValueDelimiter) {
				std::string result;

				bool first = true;
				for(const auto& item : value) {
					if(first) {
						first = false;
					} else {
						result += itemDelimiter;
					}

					result += toString(item.first) + keyValueDelimiter + toString(item.second);
				}

				return result;
			}

			/**
             * Splits a string to a vector.
             * @param value The string to split.
             * @param pairDelimiter The delimiter that delimits two vector items.
             * @param trim Whether to trim the values in the vector.
             * @return A string representation.
             */
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

			/**
             * Splits a string to a map.
             * @param value The string to split.
             * @param itemDelimiter The delimiter that delimits two map items.
             * @param keyValueDelimiter The delimiter that delimits a key and a value.
             * @param trim Whether to trim the values in the map.
             * @return A string representation.
             */
			static std::map<std::string, std::string> splitToMap(const std::string& value, const std::string& itemDelimiter, const std::string& keyValueDelimiter, const bool& trim = false) {
				// First extract the items
				std::vector<std::string> items = splitToVector(value, itemDelimiter, trim);

				// Then, for each item, extract the key and value and add them to the result
				std::map<std::string, std::string> result;
				for(const auto& item : items) {
					std::vector<std::string> keyValue = splitToVector(item, keyValueDelimiter, trim);
					result[keyValue[0]] = keyValue[1];
				}

				return result;
			}

			/**
             * Formats a timestamp value.
             * @param timestamp The timestamp.
             * @param format The format to use.
             * @return The formatted timestamp.
             */
			static std::string formatTimestamp(const std::time_t& timestamp, const std::string& format = "%Y-%m-%d %H:%M:%S") {
				// Get the current date and time
				auto localTime = *std::localtime(&timestamp);
				std::ostringstream outputStringStream;
				outputStringStream << std::put_time(&localTime, format.c_str());

				return outputStringStream.str();
			}

			/**
             * Formats a timestamp value.
             * @param timestamp The timestamp.
             * @param format The format to use.
             * @return The formatted timestamp.
             */
			static std::string formatTimestamp(const std::chrono::system_clock::time_point& timestamp, std::string format = "%Y-%m-%d %H:%M:%S.%Ms") {
				format = format.replace(format.find("%Ms"), 3, toString((std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()) % std::chrono::seconds(1)).count()));
				return formatTimestamp(std::chrono::system_clock::to_time_t(timestamp), format);
			}

			/**
             * Converts a duration to a string.
             * @param duration The duration.
             * @return A string representation.
             */
			static std::string formatDuration(const std::chrono::system_clock::duration& value) {
				using Days = std::chrono::duration<int, std::ratio<86400>>;

				auto durationLeft = value;
				std::vector<std::string> result;

				const auto days = std::chrono::duration_cast<Days>(durationLeft);
				if(days.count() > 0) {
					durationLeft -= days;
					result.push_back(toString(days.count()) + " days");
				}

				const auto hours = std::chrono::duration_cast<std::chrono::hours>(durationLeft);
				if(hours.count() > 0) {
					durationLeft -= hours;
					result.push_back(toString(hours.count()) + " hours");
				}

				const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(durationLeft);
				if(minutes.count() > 0) {
					durationLeft -= minutes;
					result.push_back(toString(minutes.count()) + " minutes");
				}

				const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(durationLeft);
				if(seconds.count() > 0) {
					durationLeft -= seconds;
					result.push_back(toString(seconds.count()) + " seconds");
				}

				const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(durationLeft);
				if(milliseconds.count() > 0) {
					durationLeft -= milliseconds;
					result.push_back(toString(milliseconds.count()) + " milliseconds");
				}

				const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(durationLeft);
				if(microseconds.count() > 0) {
					durationLeft -= microseconds;
					result.push_back(toString(microseconds.count()) + " microseconds");
				}

				const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(durationLeft);
				if(nanoseconds.count() > 0) {
					durationLeft -= nanoseconds;
					result.push_back(toString(nanoseconds.count()) + " nanoseconds");
				}

				return join(result, " ");
			}

			/**
             * Extracts an argument from a set of arguments.
             * @param arguments The argument set.
             * @param index The index of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<typename Type>
			static Type getArgument(const std::vector<std::string>& arguments, const size_t& index, const Type& defaultValue = Type()) {
				return static_cast<Type>(arguments.size() > index ? arguments[index] : defaultValue);
			}

			/**
             * Extracts a string argument from a set of arguments.
             * @param arguments The argument set.
             * @param index The index of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<typename Type>
			static std::string getArgument(const std::vector<std::string>& arguments, const size_t& index, const std::string& defaultValue) {
				return arguments.size() > index ? arguments[index] : defaultValue;
			}

			/**
             * Extracts an argument from a set of arguments.
             * @param arguments The argument set.
             * @param index The index of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			std::string getArgument(const std::vector<std::string>& arguments, const size_t& index, const std::string& defaultValue) {
				return arguments.size() > index ? arguments[index] : defaultValue;
			}

			/**
             * Extracts an int argument from a set of arguments.
             * @param arguments The argument set.
             * @param index The index of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			int getArgument(const std::vector<std::string>& arguments, const size_t& index, const int& defaultValue) {
				return arguments.size() > index ? std::stoi(arguments[index]) : defaultValue;
			}

			/**
             * Extracts an unsigned int argument from a set of arguments.
             * @param arguments The argument set.
             * @param index The index of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			unsigned int getArgument(const std::vector<std::string>& arguments, const size_t& index, const unsigned int& defaultValue) {
				return arguments.size() > index ? std::stoul(arguments[index]) : defaultValue;
			}

			/**
             * Extracts a long argument from a set of arguments.
             * @param arguments The argument set.
             * @param index The index of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			long getArgument(const std::vector<std::string>& arguments, const size_t& index, const long& defaultValue) {
				return arguments.size() > index ? std::stol(arguments[index]) : defaultValue;
			}

			/**
             * Extracts an unsigned long argument from a set of arguments.
             * @param arguments The argument set.
             * @param index The index of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			unsigned long getArgument(const std::vector<std::string>& arguments, const size_t& index, const unsigned long& defaultValue) {
				return arguments.size() > index ? std::stoul(arguments[index]) : defaultValue;
			}

			/**
             * Extracts a duration argument from a set of arguments.
             * @param arguments The argument set.
             * @param index The index of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			std::chrono::system_clock::duration getArgument(const std::vector<std::string>& arguments, const size_t& index, const std::chrono::system_clock::duration& defaultValue) {
				return arguments.size() > index ? std::chrono::milliseconds(std::stoul(arguments[index])) : defaultValue;
			}

			/**
             * Extracts an argument from a set of arguments.
             * @param arguments The argument set.
             * @param name The name of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<typename Type>
			static Type getArgument(const std::map<std::string, std::string>& arguments, const std::string& name, const Type& defaultValue = Type()) {
				return arguments.find(name) != arguments.end() ? static_cast<Type>(arguments.at(name)) : defaultValue;
			}

			/**
             * Extracts a string argument from a set of arguments.
             * @param arguments The argument set.
             * @param name The name of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			std::string getArgument(const std::map<std::string, std::string>& arguments, const std::string& name, const std::string& defaultValue) {
				return arguments.find(name) != arguments.end() ? arguments.at(name) : defaultValue;
			}

			/**
             * Extracts a bool argument from a set of arguments.
             * @param arguments The argument set.
             * @param name The name of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			bool getArgument(const std::map<std::string, std::string>& arguments, const std::string& name, const bool& defaultValue) {
				return arguments.find(name) != arguments.end() ? std::stoi(arguments.at(name)) : defaultValue;
			}

			/**
             * Extracts an int argument from a set of arguments.
             * @param arguments The argument set.
             * @param name The name of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			int getArgument(const std::map<std::string, std::string>& arguments, const std::string& name, const int& defaultValue) {
				return arguments.find(name) != arguments.end() ? std::stoi(arguments.at(name)) : defaultValue;
			}

			/**
             * Extracts an unsigned int argument from a set of arguments.
             * @param arguments The argument set.
             * @param name The name of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			unsigned int getArgument(const std::map<std::string, std::string>& arguments, const std::string& name, const unsigned int& defaultValue) {
				return arguments.find(name) != arguments.end() ? std::stoul(arguments.at(name)) : defaultValue;
			}

			/**
             * Extracts a long argument from a set of arguments.
             * @param arguments The argument set.
             * @param name The name of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			long getArgument(const std::map<std::string, std::string>& arguments, const std::string& name, const long& defaultValue) {
				return arguments.find(name) != arguments.end() ? std::stol(arguments.at(name)) : defaultValue;
			}

			/**
             * Extracts an unsigned long argument from a set of arguments.
             * @param arguments The argument set.
             * @param name The name of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			unsigned long getArgument(const std::map<std::string, std::string>& arguments, const std::string& name, const unsigned long& defaultValue) {
				return arguments.find(name) != arguments.end() ? std::stoul(arguments.at(name)) : defaultValue;
			}

			/**
             * Extracts a duration argument from a set of arguments.
             * @param arguments The argument set.
             * @param name The name of the argument to extract.
             * @param defaultValue The default value to use if it does not exist.
             * @return The argument's value.
             */
			template<>
			std::chrono::system_clock::duration getArgument(const std::map<std::string, std::string>& arguments, const std::string& name, const std::chrono::system_clock::duration& defaultValue) {
				return arguments.find(name) != arguments.end() ? std::chrono::milliseconds(std::stoul(arguments.at(name))) : defaultValue;
			}

			/**
             * Flattens a map into a vector.
             * @tparam KeyType The type of the keys.
             * @tparam ValueType The type of the values.
             * @param value The map.
             * @return The vector.
             */
			template<typename KeyType, typename ValueType>
			static std::vector<std::string> flatten(const std::map<KeyType, ValueType>& value) {
				std::vector<std::string> result;

				for(const auto& mapElement : value) {
					result.push_back(toString(mapElement.first));
					result.push_back(toString(mapElement.second));
				}

				return result;
			}

			/**
             * Parses the parameters to a vector.
             * @param argumentCount The amount of arguments.
             * @param argumentValues The argument values.
             * @return The argument vector.
             */
			static std::vector<std::string> parseArgumentsVector(const unsigned int& argumentCount, char* argumentValues[]) {
				return std::vector<std::string>(argumentValues + 1, argumentValues + argumentCount);
			}

			/**
             * Parses the parameters to a vector.
             * @param argumentCount The amount of arguments.
             * @param argumentValues The argument values.
             * @return The argument vector.
             */
			static std::map<std::string, std::string> parseArgumentsMap(const unsigned int& argumentCount, char* argumentValues[]) {
				std::map<std::string, std::string> results;

				if(argumentCount > 0) {
					std::string argumentName = argumentValues[0];

					for(unsigned int index = 1; index < argumentCount; ++index) {
						// Check if we are processing a name or value
						if(argumentValues[index][0] == '-') {
							// Processing a name
							argumentName = argumentValues[index];

							// Set the default value
							results[argumentName] = "1";
						} else {
							// Processing a value
							results[argumentName] = argumentValues[index];
						}
					}
				}

				return results;
			}

			/**
             * Serializes a map of arguments.
             * @param arguments The arguments.
             * @return The serialized arguments.
             */
			static std::string serializeArgumentsMap(const std::map<std::string, std::string>& arguments) {
				return join(arguments, " ", " ");
			}

			/**
			 * Reads the entire contents of a file.
			 * @param path The file to read.
			 * @return The contents.
			 */
			static std::string readFile(const std::string& path) {
				std::ifstream inputStream(path);
				return std::string(std::istreambuf_iterator<char>(inputStream), std::istreambuf_iterator<char>());
			}

			/**
			 * Writes a file.
			 * @param path The path to the file.
			 * @param value The value to write.
			 */
			static void writeFile(const std::string& path, const std::string& value) {
				std::ofstream maximumRateStream(path);
				maximumRateStream << value;
			}

			/**
			 * Parses a table file.
			 * Assumes the first row is the header row.
			 * @param table The table to parse.
			 * @param rowDelimiter The delimiter between rows.
			 * @param columnDelimiter The delimiter between columns.
			 * @return The rows.
			 */
			static std::vector<std::map<std::string, std::string>> parseTable(const std::string& table, const std::string& rowDelimiter = "\n", const std::string& columnDelimiter = ",") {
				std::vector<std::map<std::string, std::string>> result = {};

				const auto rows = splitToVector(table, rowDelimiter);
				std::vector<std::string> headers = {};
				for(const auto& row : rows) {
					// Only process rows that contain data
					if(!row.empty()) {
						const auto cells = splitToVector(row, columnDelimiter);

						// Set the headers if not yet set
						if(headers.empty()) {
							headers = cells;
						} else {
							// Add the row
							result.emplace_back();

							// Add the cells
							for(unsigned int cellIndex = 0; cellIndex < cells.size(); ++cellIndex) {
								const auto& header = headers[cellIndex];
								const auto& cell = cells[cellIndex];

								result.back()[header] = cell;
							}
						}
					}
				}

				return result;
			}
		}
	}
}