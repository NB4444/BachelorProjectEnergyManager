#include "./Application.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <memory>
#include <sched.h>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace EnergyManager {
	Application::Application(std::string path) : path_(std::move(path)) {
	}

	Application::Application(const Application& application) : Application(application.path_) {
	}

	bool Application::isRunning() const {
		return isRunning_;
	}

	std::vector<std::shared_ptr<Hardware::CPU>> Application::getCPUAffinity() const {
		// Get the affinity mask
		cpu_set_t cpuMask;
		if(sched_getaffinity(processID_, sizeof(cpuMask), &cpuMask) != 0) {
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not get affinity");
		}

		// Decode the mask
		std::vector<std::shared_ptr<Hardware::CPU>> cpus;
		for(const auto& cpu : Hardware::CPU::getCPUs()) {
			if(CPU_ISSET(cpu->getID(), &cpuMask)) {
				cpus.push_back(cpu);
			}
		}

		return cpus;
	}

	void Application::setCPUAffinity(const std::vector<std::shared_ptr<Hardware::CPU>>& affinity) {
		// Encode an affinity mask
		cpu_set_t mask;
		CPU_ZERO(&mask);
		for(const auto& cpu : affinity) {
			CPU_SET(cpu->getID(), &mask);
		}

		// Set the affinity
		if(sched_setaffinity(processID_, sizeof(mask), &mask) != 0) {
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not set affinity");
		}
	}

	std::string Application::getExecutableOutput() const {
		return executableOutput_;
	}

	void Application::run(std::vector<std::string> parameters, const std::vector<std::shared_ptr<Hardware::CPU>>& cpuAffinity) {
		if(isRunning_) {
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Application is already running!");
		}

		// Define required constants
		const bool readOnly = true;
		const unsigned int readPipe = 0;
		const unsigned int writePipe = 1;

		// Transform parameters
		std::vector<char*> cStringParameters;
		std::transform(parameters.begin(), parameters.end(), std::back_inserter(cStringParameters), [](auto& parameter) {
			return const_cast<char*>(parameter.c_str());
		});

		// Keep track of application status
		isRunning_ = true;
		executableOutput_ = "";

		// Fork the current process and open a set of pipes
		int pipes[2];
		pipe(pipes);
		if((processID_ = fork()) == -1) {
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not fork current process");
		}
		bool isParent = processID_ > 0;

		// Set up the pipes
		if(isParent) {
			if(readOnly) {
				close(pipes[writePipe]);
			} else {
				close(pipes[readPipe]);
			}
		} else {
			// Set up the pipes
			if(readOnly) {
				close(pipes[readPipe]);
				dup2(pipes[writePipe], 1);
			} else {
				close(pipes[writePipe]);
				dup2(pipes[readPipe], 0);
			}

			// Start the child process
			setpgid(processID_, processID_);
			execvp(path_.c_str(), cStringParameters.data());
		}

		// Set the application's affinity
		if(!cpuAffinity.empty()) {
			setCPUAffinity(cpuAffinity);
		}

		// Get a pointer to capture application output
		FILE* filePointer = readOnly ? fdopen(pipes[readPipe], "r") : fdopen(pipes[writePipe], "w");

		// Retrieve the executable's output
		char lineBuffer[1024];
		while(fgets(lineBuffer, sizeof(lineBuffer), filePointer)) {
			executableOutput_ += std::string(lineBuffer);
		}

		// Clean up
		fclose(filePointer);

		isRunning_ = false;
	}
}