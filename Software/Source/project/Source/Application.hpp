#pragma once

#include "Persistence/Entity.hpp"

#include <functional>
#include <string>
#include <thread>
#include <vector>

/**
 * An executable application.
 */
class Application : public Persistence::Entity<Application> {
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

	///**
	// * The output from the energy monitor.
	// */
	//std::string cudaEnergyMonitorOutput_;

	/**
	 * The thread that monitors the executable.
	 */
	std::thread executableMonitor_;

	///**
	// * The thread that monitors the CUDA energy monitor.
	// */
	//std::thread cudaEnergyMonitorMonitor_;

	std::map<std::string, std::string> onSave() override;

public:
	Application(const std::map<std::string, std::string>& row);

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

	///**
	// * Gets the CUDA Energy Monitor's output.
	// * @return The output.
	// */
	//std::string getCUDAEnergyMonitorOutput() const;

	/**
	 * Starts the Application.
	 * @param parameters The parameters to provide to the executable.
	 */
	void start(const std::vector<std::string>& parameters);

	/**
	 * Waits until the Application has finished running.
	 */
	void waitUntilDone();
};