#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <stdarg.h>
#include <string>

namespace Utility {
	namespace Logging {
		static void vlog(const std::string& header, std::string format, va_list& arguments) {
			// Get the current date and time
			auto time = std::time(nullptr);
			auto localTime = *std::localtime(&time);
			std::ostringstream outputStringStream;
			outputStringStream << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");

			// Prepend header
			format = "[" + outputStringStream.str() + "] [" + header + "] " + format + '\n';

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