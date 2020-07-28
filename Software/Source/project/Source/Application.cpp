#include "Application.hpp"

#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

std::string Application::execute(const std::filesystem::path& path, const std::vector<std::string>& parameters, std::function<std::string(const std::string&)>& dataHandler) {
	std::string result;

	// Generate the execute command
	std::string command = path.generic_string() + " " + std::accumulate(parameters.begin(), parameters.end(), std::string(), [](const std::string& left, const std::string& right) -> std::string {
							  return left + (left.length() > 0 ? " " : "") + right;
						  });

	// Start and get a pointer to the running executable
	std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);

	// Check if the executable was started
	if(!pipe) {
		throw std::runtime_error("Failed to start application");
	}

	// Retrieve the executable's output
	std::array<char, 128> buffer {};
	while(fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
		result += dataHandler(buffer.data());
	}

	return result;
}

Application::Application(std::filesystem::path path)
	: path_(std::move(path)) {
}

std::string Application::start(const std::vector<std::string>& parameters) const {
	bool running = true;

	// Start the executable
	std::string executableCommand = "\"" + path_.generic_string() + "\" " + std::accumulate(parameters.begin(), parameters.end(), std::string(), [](const std::string& left, const std::string& right) -> std::string {
										return left + (left.length() > 0 ? " " : "") + right;
									});
	std::unique_ptr<FILE, decltype(&pclose)> executablePipe(popen(executableCommand.c_str(), "r"), pclose);
	if(!executablePipe) {
		throw std::runtime_error("Failed to start application");
	}
	std::string executableResult;
	std::thread executableMonitor([&] {
		std::cout << "[EXECUTABLE] Starting executable..." << std::endl;

		// Retrieve the executable's output
		std::array<char, 128> buffer {};
		while(fgets(buffer.data(), buffer.size(), executablePipe.get()) != nullptr) {
			executableResult += buffer.data();
			std::cout << "[EXECUTABLE] " << buffer.data() << std::endl;
		}

		std::cout << "[EXECUTABLE] Stopping executable..." << std::endl;
	});

	// Start the energy monitor
	std::string energyMonitorCommand = "\"" + std::string(CUDA_ENERGY_MONITOR_EXECUTABLE) + "\"";
	std::unique_ptr<FILE, decltype(&pclose)> cudaEnergyMonitorPipe(popen(energyMonitorCommand.c_str(), "r"), pclose);
	if(!cudaEnergyMonitorPipe) {
		throw std::runtime_error("Failed to start CUDA energy monitor");
	}
	std::string cudaEnergyMonitorResult;
	std::thread cudaEnergyMonitorMonitor([&] {
		std::cout << "[CUDA ENERGY MONITOR] Starting CUDA energy monitory..." << std::endl;

		// Retrieve the executable's output
		std::array<char, 128> buffer {};
		while(running && fgets(buffer.data(), buffer.size(), cudaEnergyMonitorPipe.get()) != nullptr) {
			cudaEnergyMonitorResult += buffer.data();
			std::cout << "[CUDA ENERGY MONITOR] " << buffer.data() << std::endl;
		}

		std::cout << "[CUDA ENERGY MONITOR] Stopping CUDA energy monitory..." << std::endl;
	});

	// Wait for the process to finish
	executableMonitor.join();

	// Terminate the energy monitor
	running = false;
	cudaEnergyMonitorMonitor.join();

	return executableResult;
}