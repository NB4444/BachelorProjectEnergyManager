#pragma once

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace EnergyManager {
	namespace Utility {
		namespace Environment {
			/**
			 * Gets an environment variable.
			 * @tparam Type The type of the variable.
			 * @param name The name of the variable.
			 * @return The variable value.
			 */
			template<typename Type>
			static Type getVariable(const std::string& name) {
				return static_cast<Type>(getenv(name.c_str()));
			}

			/**
			 * Gets an environment variable string.
			 * @param name The name of the variable.
			 * @return The variable value.
			 */
			template<>
			std::string getVariable(const std::string& name) {
				return getVariable<char*>(name);
			}

			/**
			 * Gets an environment variable unsigned integer.
			 * @param name The name of the variable.
			 * @return The variable value.
			 */
			template<>
			unsigned int getVariable(const std::string& name) {
				return std::stoul(getVariable<std::string>(name));
			}

			/**
			 * Sets an environment variable.
			 * @tparam Type The type of the variable.
			 * @param name The name of the variable.
			 * @param value The variable value.
			 */
			template<typename Type>
			static void setVariable(const std::string& name, const Type& value) {
				setenv(name.c_str(), Text::toString(value).c_str(), 1);
			}

			/**
			 * Gets the path to the current application.
			 * @return The path.
			 */
			static std::string getApplicationPath() {
				// Cache the path
				static std::string path = [] {
					// Get the path to the application
					char applicationPathBuffer[PATH_MAX];
					if(readlink("/proc/self/exe", applicationPathBuffer, PATH_MAX) < 0) {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Invalid application path");
					}

					return std::string(applicationPathBuffer);
				}();

				return path;
			}

			/**
			 * Gets the current hostname.
			 * @return The hostname.
			 */
			static std::string getHostname() {
				char hostname[BUFSIZ];
				gethostname(hostname, BUFSIZ);

				return hostname;
			}

			/**
			 * Checks if a file exists.
			 * @param path The path to the file.
			 * @return Whether the file exists.
			 */
			static bool fileExists(const std::string& path) {
				struct stat buffer {};
				return (stat(path.c_str(), &buffer) == 0);
			}
		}
	}
}