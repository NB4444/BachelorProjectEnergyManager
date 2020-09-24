#include "./ParseException.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Exceptions {
			ParseException::ParseException(const std::string& file, const size_t& line) : Exception("Failed to parse", file, line) {
			}
		}
	}
}