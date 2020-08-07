#include "./Application.hpp"

#include "Utility/Logging.hpp"
#include "Utility/Serialization.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

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

//std::string Application::getCUDAEnergyMonitorOutput() const {
//	return cudaEnergyMonitorOutput_;
//}

void Application::start(const std::vector<std::string>& parameters) {
	if(isRunning_) {
		throw std::runtime_error("Application is already running!");
	}

	isRunning_ = true;
	executableOutput_ = "";
	//cudaEnergyMonitorOutput_ = "";

	// Start the executable
	executableMonitor_ = std::thread([&] {
		Utility::Logging::logInformation("Starting executable...");

		std::string executableCommand = /*"LD_LIBRARY_PATH=\"" + std::string(CUDA_ENERGY_MONITOR_DIRECTORY) + "\" LD_PRELOAD=libmonitor.so "*/ "\"" + path_ + "\" " + Utility::Serialization::serialize(parameters, " ");
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
	});

	//// Start the energy monitor
	//cudaEnergyMonitorMonitor_ = std::thread([&] {
	//	Utility::Logging::logInformation("Starting CUDA energy monitor...");
	//
	//	std::string energyMonitorCommand = "\"" + std::string(CUDA_ENERGY_MONITOR_DIRECTORY) + "/monitor\"";
	//	std::unique_ptr<FILE, decltype(&pclose)> cudaEnergyMonitorPipe(popen(energyMonitorCommand.c_str(), "r"), pclose);
	//	if(!cudaEnergyMonitorPipe) {
	//		throw std::runtime_error("Failed to start CUDA energy monitor");
	//	}
	//
	//	// Retrieve the executable's output
	//	std::array<char, 256> buffer {};
	//	while(isRunning_ && fgets(buffer.data(), buffer.size(), cudaEnergyMonitorPipe.get()) != nullptr) {
	//		cudaEnergyMonitorOutput_ += buffer.data();
	//		Utility::Logging::logInformation("Processing buffer data: %s", buffer.data());
	//	}
	//
	//	Utility::Logging::logInformation("Stopping CUDA energy monitor...");
	//});
}

void Application::waitUntilDone() {
	// Wait for the processes to finish
	executableMonitor_.join();
	//cudaEnergyMonitorMonitor_.join();
}