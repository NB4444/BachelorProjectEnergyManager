#include "./Application.hpp"

#include "EnergyManager/Utility/Logging.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace EnergyManager {
	Application::Application(std::string path)
		: path_(std::move(path)) {
	}

	Application::Application(const Application& application)
		: Application(application.path_) {
	}

	bool Application::isRunning() const {
		return isRunning_;
	}

	std::string Application::getExecutableOutput() const {
		return executableOutput_;
	}

	void Application::run(const std::vector<std::string>& parameters) {
		if(isRunning_) {
			throw std::runtime_error("Application is already running!");
		}

		isRunning_ = true;
		executableOutput_ = "";

		// Start the executable
		Utility::Logging::logInformation("Starting executable...");

		std::string executableCommand = "\"" + path_ + "\" " + Utility::Text::join(parameters, " ");
		std::unique_ptr<FILE, decltype(&pclose)> executablePipe(popen(executableCommand.c_str(), "r"), pclose);
		if(!executablePipe) {
			throw std::runtime_error("Failed to start application");
		}

		// Retrieve the executable's output
		std::array<char, 256> buffer {};
		while(fgets(buffer.data(), buffer.size(), executablePipe.get()) != nullptr) {
			executableOutput_ += buffer.data();
			Utility::Logging::logInformation("Processing buffer data: %s", buffer.data());
		}

		Utility::Logging::logInformation("Stopping executable...");

		isRunning_ = false;
	}
}