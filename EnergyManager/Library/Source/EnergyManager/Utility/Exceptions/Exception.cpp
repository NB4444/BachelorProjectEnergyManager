#include "./Exception.hpp"

#include "EnergyManager/Utility/Logging.hpp"

#include <unistd.h>
#include <utility>

namespace EnergyManager {
	namespace Utility {
		namespace Exceptions {
			Exception::Exception(const std::string& message, std::string file, const size_t& line) : std::runtime_error(message), file_(std::move(file)), line_(line) {
			}

			std::string Exception::getMessage() const {
				return what();
			}

			std::string Exception::getFile() const {
				return file_;
			}

			size_t Exception::getLine() const {
				return line_;
			}

			void Exception::log() const {
				Logging::logError(getMessage(), getFile(), getLine());
			}
		}
	}
}