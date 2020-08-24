#pragma once

#include <stdexcept>

#define ENERGY_MANAGER_UTILITY_EXCEPTION(MESSAGE) throw EnergyManager::Utility::Exception(MESSAGE, __FILE__, __LINE__);

namespace EnergyManager {
	namespace Utility {
		class Exception :
			public std::runtime_error {
				std::string file_;

				size_t line_;

			public:
				Exception(const std::string& message, const std::string& file, const size_t& line);

				std::string getMessage() const;

				std::string getFile() const;

				size_t getLine() const;
		};
	}
}