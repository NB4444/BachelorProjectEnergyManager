#pragma once

#include "EnergyManager/Persistence/Entity.hpp"
#include "EnergyManager/Hardware/CPU.hpp"

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

		/**
		 * The pipe of the current executable.
		 */
		 pid_t processID_;

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
		 * Retrieves the Application's affinity.
		 * @return The affinity.
		 */
		std::vector<std::shared_ptr<Hardware::CPU>> getCPUAffinity() const;

		/**
		 * Sets the Application's affinity.
		 * @param affinity The affinity.
		 */
		void setCPUAffinity(const std::vector<std::shared_ptr<Hardware::CPU>>& affinity);

		/**
		 * Gets the Application's output.
		 * @return The output.
		 */
		std::string getExecutableOutput() const;

		/**
		 * Starts the Application.
		 * @param parameters The parameters to provide to the executable.
		 * @param cpuAffinity The CPU affinity.
		 */
		void run(std::vector<std::string> parameters, const std::vector<std::shared_ptr<Hardware::CPU>>& cpuAffinity = {});
	};
}