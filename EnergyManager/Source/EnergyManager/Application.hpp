#pragma once

#include "EnergyManager/Persistence/Entity.hpp"

#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace EnergyManager {
	/**
	 * An executable application.
	 */
	class Application {
			/**
			 * The path to the Application's main executable.
			 */
			std::string path_;

			/**
			 * Whether the Application is running.
			 */
			bool isRunning_ = false;

			/**
			 * The output from the executable.
			 */
			std::string executableOutput_;

		public:
			/**
			 * Creates a new Application.
			 * @param path The path to the Application's main executable.
			 */
			Application(std::string path);

			/**
			 * Copies the Application.
			 * @param application The Application to copy.
			 */
			Application(const Application& application);

			/**
			 * Determines if the Application is running.
			 * @return Whether the Application is running.
			 */
			bool isRunning() const;

			/**
			 * Gets the Application's output.
			 * @return The output.
			 */
			std::string getExecutableOutput() const;

			/**
			 * Starts the Application.
			 * @param parameters The parameters to provide to the executable.
			 */
			void run(const std::vector<std::string>& parameters);
	};
}