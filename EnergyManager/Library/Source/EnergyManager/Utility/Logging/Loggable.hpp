#pragma once

#include <string>
#include <vector>

namespace EnergyManager {
	namespace Utility {
		namespace Logging {
			enum class Level;

			/**
			 * Represents an object that can log information.
			 */
			class Loggable {
			protected:
				/**
				 * Generates the headers for logging.
				 * @return The headers.
				 */
				virtual std::vector<std::string> generateHeaders() const;

			public:
				/**
				 * Creates a new Loggable.
				 */
				Loggable() = default;

				/**
				 * Gets the headers used for logging.
				 * @return The headers.
				 */
				std::vector<std::string> getHeaders() const;

				/**
				 * Logs a message with a variable number of parameters.
				 * @param level The logging level.
				 * @param headers The log headers.
				 * @param format The format of the message.
				 * @param ... The arguments to use.
				 */
				void log(const Level& level, const std::vector<std::string>& headers, std::string format, ...) const;

				/**
				 * Logs a trace message.
				 * @param format The format of the message.
				 * @param ... The arguments to use.
				 */
				void logTrace(std::string format, ...) const;

				/**
				 * Logs a debug message.
				 * @param format The format of the message.
				 * @param ... The arguments to use.
				 */
				void logDebug(std::string format, ...) const;

				/**
				 * Logs an informational message.
				 * @param format The format of the message.
				 * @param ... The arguments to use.
				 */
				void logInformation(std::string format, ...) const;

				/**
				 * Logs a warning.
				 * @param format The format of the message.
				 * @param ... The arguments to use.
				 */
				void logWarning(std::string format, ...) const;

				/**
				 * Logs an error.
				 * @param format The format of the message.
				 * @param file The file in which the error occurred.
				 * @param line The line on which the error occurred.
				 * @param ... The arguments to use.
				 */
				void logError(std::string format, std::string file, int line, ...) const;
			};
		}
	}
}