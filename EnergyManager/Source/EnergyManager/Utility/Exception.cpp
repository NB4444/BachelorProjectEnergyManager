#include "./Exception.hpp"

namespace EnergyManager {
	namespace Utility {
		Exception::Exception(const std::string& message, const std::string& file, const size_t& line) : std::runtime_error(message), file_(file), line_(line) {
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
	}
}