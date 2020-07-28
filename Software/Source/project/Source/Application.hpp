#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

/**
 * An executable application.
 */
class Application {
	/**
	 * Executes an executable.
	 * @param path The path to the executable.
	 * @param parameters The parameters to provide to the executable.
	 * @return The output of the executable.
	 */
	static std::string execute(const std::filesystem::path& path, const std::vector<std::string>& parameters, std::function<std::string(const std::string&)>& dataHandler);

	/**
	 * The path to the Application's main executable.
	 */
	std::filesystem::path path_;

public:
	/**
	 * Creates a new Application.
	 * @param path The path to the Application's main executable.
	 */
	Application(std::filesystem::path path);

	/**
	 * Starts the Application.
	 * @param parameters The parameters to provide to the executable.
	 * @return The output of the executable.
	 */
	std::string start(const std::vector<std::string>& parameters) const;
};