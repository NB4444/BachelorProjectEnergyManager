#include "./Application.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace EnergyManager {
	Application::Application(std::string path) : path_(std::move(path)) {
	}

	Application::Application(const Application& application) : Application(application.path_) {
	}

	bool Application::isRunning() const {
		return isRunning_;
	}

	std::string Application::getExecutableOutput() const {
		return executableOutput_;
	}

	void Application::run(const std::vector<std::string>& parameters) {
		if(isRunning_) {
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Application is already running!");
		}

		isRunning_ = true;
		executableOutput_ = "";

		// Start the executable
		std::string executableCommand = "\"" + path_ + "\" " + Utility::Text::join(parameters, " ");
		std::unique_ptr<FILE, decltype(&pclose)> executablePipe(popen(executableCommand.c_str(), "r"), pclose);
		if(!executablePipe) {
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Failed to start application");
		}

		// Retrieve the executable's output
		std::array<char, 256> buffer {};
		while(fgets(buffer.data(), buffer.size(), executablePipe.get()) != nullptr) {
			executableOutput_ += buffer.data();
		}

		isRunning_ = false;
	}
}