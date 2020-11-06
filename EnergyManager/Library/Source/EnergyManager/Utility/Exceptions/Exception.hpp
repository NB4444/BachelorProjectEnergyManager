#pragma once

#include <csignal>
#include <execinfo.h>
#include <stdexcept>
#include <string>
#include <vector>

#define ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION(MESSAGE) throw EnergyManager::Utility::Exceptions::Exception(MESSAGE, __FILE__, __LINE__);

#define ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(STATEMENT) \
	try { \
		STATEMENT; \
	} catch(const EnergyManager::Utility::Exceptions::Exception& exception) { \
	} catch(const std::exception& exception) { \
	}

namespace EnergyManager {
	namespace Utility {
		namespace Exceptions {
			/**
			 * Represents an error that occurred.
			 */
			class Exception : public std::runtime_error {
				/**
				 * The source file in which the error occurred.
				 */
				std::string file_;

				/**
				 * The line on which the error occurred.
				 */
				size_t line_;

			public:
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
				 * Logs the error.
				 */
				void log() const;
			};
		}
	}
}