#pragma once

#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Logging/Loggable.hpp"
#include "EnergyManager/Utility/StaticInitializer.hpp"

#include <csignal>
#include <execinfo.h>
#include <stdexcept>
#include <string>
#include <vector>
#define BOOST_STACKTRACE_USE_ADDR2LINE
#include <boost/exception/all.hpp>
#include <boost/stacktrace.hpp>

#define ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION(MESSAGE) throw EnergyManager::Utility::Exceptions::Exception(MESSAGE, __FILE__, __LINE__);

#define ENERGY_MANAGER_UTILITY_EXCEPTIONS_ASSERT(CONDITION, MESSAGE) \
	if(!(CONDITION)) \
		ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION(MESSAGE);

#define ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(STATEMENT) \
	try { \
		STATEMENT; \
	} catch(const EnergyManager::Utility::Exceptions::Exception& exception) { \
	} catch(const std::exception& exception) { \
	} catch(...) { \
	}

namespace EnergyManager {
	namespace Utility {
		namespace Exceptions {
			/**
			 * Represents an error that occurred.
			 */
			class Exception
				: public std::runtime_error
				, protected Logging::Loggable {
				typedef boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace> traced;

				static const std::string backtraceFile_;

				/**
				 * Initializes the signal handler.
				 */
				static StaticInitializer signalHandlerInitializer_;

				/**
				 * The message that describes the error.
				 */
				std::string message_;

				/**
				 * The source file in which the error occurred.
				 */
				std::string file_;

				/**
				 * The line on which the error occurred.
				 */
				size_t line_;

				/**
				 * Handles signals.
				 * @param signalNumber The signal number.
				 */
				static void signalHandler(int signalNumber);

			public:
				/**
				 * Attempts an operation multiple times.
				 * @param operation The operation.
				 * @param attempts The amount of attempts. Set to 0 to retry infinitely.
				 * @param attemptInterval The time to wait between attempts.
				 * @package attemptTimeout The maximum time an attempt can take. Set to 0 to allow an attempt to run forever.
				 */
				static void retry(
					const std::function<void()>& operation,
					const unsigned int& attempts = 0,
					const std::chrono::system_clock::duration& attemptInterval = std::chrono::system_clock::duration(0));

				/**
				 * Creates a new Exception.
				 * @param message The message that describes the error.
				 * @param file The source file in which the error occurred.
				 * @param line The line on which the error occurred.
				 */
				Exception(const std::string& message, std::string file, const size_t& line);

				/**
				 * Gets the message that describes the error.
				 * @return The message.
				 */
				std::string getMessage() const;

				/**
				 * Gets the source file in which the error occurred.
				 * @return The file.
				 */
				std::string getFile() const;

				/**
				 * Gets the line on which the error occurred.
				 * @return The line.
				 */
				size_t getLine() const;

				/**
				 * Throws the exception with a stacktrace.
				 */
				void throwWithStacktrace() const;

				/**
				 * Logs the error.
				 */
				void log() const;
			};
		}
	}
}