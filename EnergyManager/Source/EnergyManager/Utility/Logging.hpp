#pragma once

#include "EnergyManager/Utility/Text.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <stdarg.h>
#include <string>

#define ENERGY_MANAGER_UTILITY_LOGGING_LOG_ERROR(FORMAT, ...) EnergyManager::Utility::Logging::logError(FORMAT, __FILE__, __LINE__, __VA_ARGS__)

namespace EnergyManager {
	namespace Utility {
		namespace Logging {
			static void vlog(const std::string& header, std::string format, va_list& arguments) {
				// Prepend header
				format = "[" + Text::formatTimestamp(std::chrono::system_clock::now()) + "] [" + header + "] " + format + '\n';

				// Print the message
				vprintf(format.c_str(), arguments);
			}

			static void log(const std::string& header, const std::string& format, ...) {
				va_list arguments;
				va_start(arguments, format);
				vlog(header, format, arguments);
				va_end(arguments);
			}

			static void logInformation(const std::string& format, ...) {
				va_list arguments;
				va_start(arguments, format);
				vlog("INFORMATION", format, arguments);
				va_end(arguments);
			}

			static void logWarning(const std::string& format, ...) {
				va_list arguments;
				va_start(arguments, format);
				vlog("WARNING", format, arguments);
				va_end(arguments);
			}

			static void logError(const std::string& format, const std::string& file, const int& line, ...) {
				va_list arguments;
				va_start(arguments, line);
				vlog("ERROR", file + ":" + std::to_string(line) + ": " + format, arguments);
				va_end(arguments);
			}
		}
	}
}