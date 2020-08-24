#pragma once

#include <stdexcept>
#include <csignal>
#include <execinfo.h>
#include <string>
#include <vector>

#define ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION(MESSAGE) throw EnergyManager::Utility::Exceptions::Exception(MESSAGE, __FILE__, __LINE__);

namespace EnergyManager {
	namespace Utility {
		namespace Exceptions {
			class Exception :
				public std::runtime_error {
					static void backtraceSignalHandler(int signal, siginfo_t* info, void* secret);

					std::string file_;

					size_t line_;

					std::vector<std::pair<std::string, std::string>> stackTrace_ = {};

				public:
					static void initialize();

					static void logGDBStackTrace();

					Exception(const std::string& message, const std::string& file, const size_t& line);

					std::string getMessage() const;

					std::string getFile() const;

					size_t getLine() const;

					void log() const;
			};
		}
	}
}