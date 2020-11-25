#include "./Loggable.hpp"

#include "EnergyManager/Utility/Logging.hpp"

#include <cstdarg>
#include <utility>

namespace EnergyManager {
	namespace Utility {
		namespace Logging {
			std::vector<std::string> Loggable::generateHeaders() const {
				return {};
			}

			std::vector<std::string> Loggable::getHeaders() const {
				return generateHeaders();
			}

			void Loggable::log(const Level& level, const std::vector<std::string>& headers, std::string format, ...) const {
				auto allHeaders = getHeaders();
				allHeaders.insert(allHeaders.end(), headers.begin(), headers.end());

				va_list arguments;
				va_start(arguments, format);
				vlog(level, allHeaders, format, arguments);
				va_end(arguments);
			}

			void Loggable::logDebug(std::string format, ...) const {
				va_list arguments;
				va_start(arguments, format);
				vlog(Level::DEBUG, getHeaders(), format, arguments);
				va_end(arguments);
			}

			void Loggable::logInformation(std::string format, ...) const {
				va_list arguments;
				va_start(arguments, format);
				vlog(Level::INFORMATION, getHeaders(), format, arguments);
				va_end(arguments);
			}

			void Loggable::logWarning(std::string format, ...) const {
				va_list arguments;
				va_start(arguments, format);
				vlog(Level::WARNING, getHeaders(), format, arguments);
				va_end(arguments);
			}

			void Loggable::logError(std::string format, std::string file, int line, ...) const {
				va_list arguments;
				va_start(arguments, line);
				vlog(Level::ERROR, getHeaders(), file + ":" + std::to_string(line) + ": " + format, arguments);
				va_end(arguments);
			}
		}
	}
}
