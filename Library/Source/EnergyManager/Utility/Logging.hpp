#pragma once

#include "EnergyManager/Utility/Logging/Loggable.hpp"
#include "EnergyManager/Utility/Runnable.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <chrono>
#include <cstdarg>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>

namespace EnergyManager {
	namespace Utility {
		namespace Logging {
			/**
			 * Whether to flush all log statements immediately.
			 */
			static bool flushAll = true;

			/**
			 * Human-understandable thread IDs.
			 */
			static std::map<std::thread::id, unsigned int> threadIDs = {};

			/**
			 * The mutex that protects thread IDs.
			 */
			static std::mutex threadIDsMutex_;

			/**
			 * The sizes of each header column.
			 */
			static std::map<unsigned int, unsigned int> headerColumnSizes = {};

			/**
			 * The maximum sizes for the header columns.
			 */
			static std::map<unsigned int, unsigned int> maximumHeaderColumnSizes = {};

			/**
			 * The default maximum column size.
			 */
			static unsigned int defaultMaximumHeaderColumnSize = 25;

			/**
			 * The logging levels.
			 */
			enum class Level { TRACE, DEBUG, INFORMATION, WARNING, ERROR };

			/**
			 * Flushes all buffers.
			 */
			static void flush() {
				std::cout.flush();
				std::cerr.flush();
			}

			/**
			 * The logging levels that are enabled.
			 */
			static const std::vector<Level> enabledLogLevels = { /*Level::TRACE, Level::DEBUG,*/ Level::INFORMATION, Level::WARNING, Level::ERROR };

			/**
			 * Registers a thread so that it's ID can be used.
			 * @param threadID The thread to register.
			 */
			static void registerThread(const std::thread::id& threadID) {
				std::lock_guard<std::mutex> guard(threadIDsMutex_);

				// Ignore threads that have already been registered
				if(threadIDs.find(threadID) != threadIDs.end()) {
					return;
				}

				// Keep track of the next thread ID.
				static unsigned int nextThreadID_ = 0;

				// Register the ID
				threadIDs[threadID] = nextThreadID_++;
			}

			/**
			 * Registers a thread so that it's ID can be used.
			 * @param thread The thread to register.
			 */
			static void registerThread(const std::thread& thread) {
				registerThread(thread.get_id());
			}

			/**
			 * Gets the ID of the current thread.
			 * @return The thread ID.
			 */
			static unsigned int getCurrentThreadID() {
				registerThread(std::this_thread::get_id());

				std::lock_guard<std::mutex> guard(threadIDsMutex_);

				return threadIDs.at(std::this_thread::get_id());
			}

			/**
			 * Logs a message with a variable number of parameters.
			 * @param level The logging level.
			 * @param headers The log headers.
			 * @param format The format of the message.
			 * @param arguments The arguments to use.
			 */
			static void vlog(const Level& level, std::vector<std::string> headers, const std::string& format, va_list& arguments) {
				// Check if this log level is enabled
				if(std::find(enabledLogLevels.begin(), enabledLogLevels.end(), level) == enabledLogLevels.end()) {
					return;
				}

				// Add timestamp
				headers.insert(headers.begin(), Text::formatTimestamp(std::chrono::system_clock::now()));

				// Add thread
				headers.insert(headers.begin() + 1, "Thread " + Text::toString(getCurrentThreadID()));

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
				headers.insert(headers.begin() + 2, levelString);

				// Lock the rest of the function to ensure that messages do not overlap
				static std::mutex mutex;
				std::lock_guard<std::mutex> guard(mutex);

				// Print the headers
				if(!headers.empty()) {
					for(unsigned int headerIndex = 0; headerIndex < headers.size(); ++headerIndex) {
						auto header = headers[headerIndex];
						const auto maximumColumnWidth
							= maximumHeaderColumnSizes.find(headerIndex) == maximumHeaderColumnSizes.end() ? defaultMaximumHeaderColumnSize : maximumHeaderColumnSizes.at(headerIndex);

						// Get the column size
						if(headerColumnSizes.find(headerIndex) == headerColumnSizes.end() || headerColumnSizes.at(headerIndex) < header.length()) {
							// Column not found or smaller than the current header
							headerColumnSizes[headerIndex] = header.length() < maximumColumnWidth ? header.length() : maximumColumnWidth;
						}
						unsigned int headerColumnSize = headerColumnSizes[headerIndex];

						// Trim the header if necessary
						if(header.length() > headerColumnSize) {
							header.resize(headerColumnSize - 3);
							header += "...";
						}

						// Print the header
						printf(("%-" + Text::toString(headerColumnSize) + "s   ").c_str(), header.c_str());
					}
				}

				// Print the message
				vprintf((format + '\n').c_str(), arguments);

				// Flush if necessary
				if(flushAll) {
					flush();
				}
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
			static void logError(const std::string& format, const std::string& file, int line, ...) {
				va_list arguments;
				va_start(arguments, line);
				vlog(Level::ERROR, {}, file + ":" + std::to_string(line) + ": " + format, arguments);
				va_end(arguments);
			}
		}
	}
}