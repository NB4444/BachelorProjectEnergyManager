#pragma once

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#define ENERGY_MANAGER_UTILITY_EXCEPTIONS_PARSE_EXCEPTION() throw EnergyManager::Utility::Exceptions::ParseException(__FILE__, __LINE__);

namespace EnergyManager {
	namespace Utility {
		namespace Exceptions {
			class ParseException : public Exception {
			public:
				ParseException(const std::string& file, const size_t& line);
			};
		}
	}
}