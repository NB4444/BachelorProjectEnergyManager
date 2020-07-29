#include "Application.hpp"

#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

Application::Application(std::filesystem::path path)
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

std::string Application::getCUDAEnergyMonitorOutput() const {
	return cudaEnergyMonitorOutput_;
}

void Application::start(const std::vector<std::string>& parameters) {
	if(isRunning_) {
		throw std::runtime_error("Application is already running!");
	}

	isRunning_ = true;
	executableOutput_ = "";
	cudaEnergyMonitorOutput_ = "";

	// Start the executable
	executableMonitor_ = std::thread([&] {
		std::cout << "[EXECUTABLE] Starting executable..." << std::endl;

		std::string executableCommand = "LD_PRELOAD=\"" + std::string(CUDA_ENERGY_MONITOR_LIBRARY) + "\" \"" + path_.generic_string() + "\" " + std::accumulate(parameters.begin(), parameters.end(), std::string(), [](const std::string& left, const std::string& right) -> std::string {
											return left + (left.length() > 0 ? " " : "") + right;
										});
		std::unique_ptr<FILE, decltype(&pclose)> executablePipe(popen(executableCommand.c_str(), "r"), pclose);
		if(!executablePipe) {
			throw std::runtime_error("Failed to start application");
		}

		// Retrieve the executable's output
		std::array<char, 256> buffer {};
		while(fgets(buffer.data(), buffer.size(), executablePipe.get()) != nullptr) {
			executableOutput_ += buffer.data();
			std::cout << "[EXECUTABLE] Processing buffer data: " << buffer.data() << std::endl;
		}

		std::cout << "[EXECUTABLE] Stopping executable..." << std::endl;

		isRunning_ = false;
	});

	// Start the energy monitor
	cudaEnergyMonitorMonitor_ = std::thread([&] {
		std::cout << "[CUDA ENERGY MONITOR] Starting CUDA energy monitory..." << std::endl;

		std::string energyMonitorCommand = "\"" + std::string(CUDA_ENERGY_MONITOR_EXECUTABLE) + "\"";
		std::unique_ptr<FILE, decltype(&pclose)> cudaEnergyMonitorPipe(popen(energyMonitorCommand.c_str(), "r"), pclose);
		if(!cudaEnergyMonitorPipe) {
			throw std::runtime_error("Failed to start CUDA energy monitor");
		}

		// Retrieve the executable's output
		std::array<char, 256> buffer {};
		while(isRunning_ && fgets(buffer.data(), buffer.size(), cudaEnergyMonitorPipe.get()) != nullptr) {
			cudaEnergyMonitorOutput_ += buffer.data();
			std::cout << "[CUDA ENERGY MONITOR] Processing buffer data: " << buffer.data() << std::endl;
		}

		std::cout << "[CUDA ENERGY MONITOR] Stopping CUDA energy monitory..." << std::endl;
	});
}

void Application::waitUntilDone() {
	// Wait for the processes to finish
	executableMonitor_.join();
	cudaEnergyMonitorMonitor_.join();
}