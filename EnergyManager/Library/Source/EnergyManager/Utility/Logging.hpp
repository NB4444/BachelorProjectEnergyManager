#pragma once

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
			 * Logs a message with a variable number of parameters.
			 * @param header The log header.
			 * @param format The format of the message.
			 * @param arguments The arguments to use.
			 */
			static void vlog(const std::string& header, std::string format, va_list& arguments) {
				// Prepend header
				format = "[" + Text::formatTimestamp(std::chrono::system_clock::now()) + "] [" + header + "] " + format + '\n';

				// Print the message
				vprintf(format.c_str(), arguments);
			}

			/**
			 * Logs a message with a variable number of parameters.
			 * @param header The log header.
			 * @param format The format of the message.
			 * @param ... The arguments to use.
			 */
			static void log(const std::string& header, std::string format, ...) {
				va_list arguments;
				va_start(arguments, format);
				vlog(header, format, arguments);
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
				vlog("INFORMATION", format, arguments);
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
				vlog("WARNING", format, arguments);
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
				vlog("ERROR", file + ":" + std::to_string(line) + ": " + format, arguments);
				va_end(arguments);
			}
		}
	}
}