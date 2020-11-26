#pragma once

#include "EnergyManager/Utility/Logging/Loggable.hpp"
#include "EnergyManager/Utility/Runnable.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <chrono>
#include <cstdarg>
#include <ctime>
#include <iomanip>
#include <string>

#define ENERGY_MANAGER_UTILITY_LOGGING_LOG_ERROR(FORMAT, ...) EnergyManager::Utility::Logging::logError(FORMAT, __FILE__, __LINE__, __VA_ARGS__)

namespace EnergyManager {
	namespace Utility {
		namespace Logging {
			/**
			 * The logging levels.
			 */
			enum class Level { TRACE, DEBUG, INFORMATION, WARNING, ERROR };

			/**
			 * The logging levels that are enabled.
			 */
			static const std::vector<Level> enabledLogLevels = {
				//Level::TRACE,
				Level::DEBUG,
				Level::INFORMATION,
				Level::WARNING,
				Level::ERROR
			};

			/**
			 * Logs a message with a variable number of parameters.
			 * @param level The logging level.
			 * @param headers The log headers.
			 * @param format The format of the message.
			 * @param arguments The arguments to use.
			 */
			static void vlog(const Level& level, std::vector<std::string> headers, std::string format, va_list& arguments) {
				// Check if this log level is enabled
				if(std::find(enabledLogLevels.begin(), enabledLogLevels.end(), level) == enabledLogLevels.end()) {
					return;
				}

				// Add timestamp
				headers.insert(headers.begin(), Text::formatTimestamp(std::chrono::system_clock::now()));

				// Add level
				std::string levelString;
				switch(level) {
					case Level::TRACE:
						levelString = "TRACE";
						break;
					case Level::DEBUG:
						levelString = "DEBUG";
						break;
					default:
					case Level::INFORMATION:
						levelString = "INFORMATION";
						break;
					case Level::WARNING:
						levelString = "WARNING";
						break;
					case Level::ERROR:
						levelString = "ERROR";
						break;
				}
				headers.push_back(levelString);

				// Prepend headers
				if(!headers.empty()) {
					format = "[" + Text::join(headers, "] [") + "] " + format + '\n';
				}

				// Print the message
				vprintf(format.c_str(), arguments);
			}

			/**
			 * Logs a message with a variable number of parameters.
			 * @param level The logging level.
			 * @param headers The log headers.
			 * @param format The format of the message.
			 * @param ... The arguments to use.
			 */
			static void log(const Level& level, const std::vector<std::string>& headers, std::string format, ...) {
				va_list arguments;
				va_start(arguments, format);
				vlog(level, headers, format, arguments);
				va_end(arguments);
			}

			/**
			 * Logs a trace message.
			 * @param format The format of the message.
			 * @param ... The arguments to use.
			 */
			static void logTrace(std::string format, ...) {
				va_list arguments;
				va_start(arguments, format);
				vlog(Level::TRACE, {}, format, arguments);
				va_end(arguments);
			}

			/**
			 * Logs a debug message.
			 * @param format The format of the message.
			 * @param ... The arguments to use.
			 */
			static void logDebug(std::string format, ...) {
				va_list arguments;
				va_start(arguments, format);
				vlog(Level::DEBUG, {}, format, arguments);
				va_end(arguments);
			}

			/**
			 * Logs an informational message.
			 * @param format The format of the message.
			 * @param ... The arguments to use.
			 */
			static void logInformation(std::string format, ...) {
				va_list arguments;
				va_start(arguments, format);
				vlog(Level::INFORMATION, {}, format, arguments);
				va_end(arguments);
			}

			/**
			 * Logs a warning.
			 * @param format The format of the message.
			 * @param ... The arguments to use.
			 */
			static void logWarning(std::string format, ...) {
				va_list arguments;
				va_start(arguments, format);
				vlog(Level::WARNING, {}, format, arguments);
				va_end(arguments);
			}

			/**
			 * Logs an error.
			 * @param format The format of the message.
			 * @param file The file in which the error occurred.
			 * @param line The line on which the error occurred.
			 * @param ... The arguments to use.
			 */
			static void logError(std::string format, std::string file, int line, ...) {
				va_list arguments;
				va_start(arguments, line);
				vlog(Level::ERROR, {}, file + ":" + std::to_string(line) + ": " + format, arguments);
				va_end(arguments);
			}
		}
	}
}