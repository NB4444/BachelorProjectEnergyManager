#pragma once

#include "EnergyManager/Utility/Text.hpp"

#include <string>

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
			 * Sets an environment variable.
			 * @tparam Type The type of the variable.
			 * @param name The name of the variable.
			 * @param value The variable value.
			 */
			template<typename Type>
			static void setVariable(const std::string& name, const Type& value) {
				const auto environmentVariable = name + "=" + Text::toString(value);
				char environmentVariableCString[environmentVariable.length() + 1];
				strcpy(environmentVariableCString, environmentVariable.c_str());
				putenv(environmentVariableCString);
			}
		}
	}
}