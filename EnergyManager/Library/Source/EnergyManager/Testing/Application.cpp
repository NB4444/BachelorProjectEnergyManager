#include "./Application.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <memory>
#include <sched.h>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace EnergyManager {
	namespace Testing {
		void Application::beforeRun() {
			// Set the active GPU
			if(gpu_ != nullptr) {
				Utility::Logging::logInformation("Changing active GPU to %d...", gpu_->getID());
				gpu_->makeActive();
			}
		}

		void Application::onRun() {
			// Define required constants
			const unsigned int readPipe = 0;
			const unsigned int writePipe = 1;

			// Transform parameters
			std::vector<char*> cStringParameters;
			std::transform(parameters_.begin(), parameters_.end(), std::back_inserter(cStringParameters), [](auto& parameter) {
				return const_cast<char*>(parameter.c_str());
			});

			// Keep track of application output
			executableOutput_ = "";

			// Create communication pipes
			Utility::Logging::logInformation("Setting up pipe to application...");
			int outputPipe[2]; // Pipe to transfer output from child to parent
			if(pipe(outputPipe) == -1) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not set up output pipe");
			}
			int startPipe[2]; // Pipe to send start signal from parent to child
			if(pipe(startPipe) == -1) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not set up start pipe");
			}

			// Fork the current process and open a set of pipes
			Utility::Logging::logInformation("Forking current process...");
			if((processID_ = fork()) < 0) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not fork current process");
			}
			bool isParent = processID_ > 0;

			// Set up the pipes
			if(isParent) {
				Utility::Logging::logInformation("Configuring pipe in parent process...");
				close(outputPipe[writePipe]);
				close(startPipe[readPipe]);

				// Set the application's affinity
				if(!cpuAffinity_.empty()) {
					setCPUAffinity(cpuAffinity_);
				}

				// Signal on the start pipe
				std::string signal = "START";
				write(startPipe[writePipe], signal.c_str(), signal.size());
				close(startPipe[writePipe]);

				// Get a pointer to capture application output
				Utility::Logging::logInformation("Capturing child output...");
				FILE* filePointer = fdopen(outputPipe[readPipe], "r");
				std::array<char, 128> outputBuffer {};
				while(fgets(outputBuffer.data(), outputBuffer.size(), filePointer) != nullptr) {
					auto output = std::string(outputBuffer.data());

					Utility::Logging::logInformation("Processing child output: %s", Utility::Text::trim(output).c_str());
					executableOutput_ += output;
				}
				close(outputPipe[readPipe]);

				// Clean up
				fclose(filePointer);
			} else {
				Utility::Logging::logInformation("Configuring pipe in child process...");
				close(outputPipe[readPipe]);
				close(startPipe[writePipe]);

				// Wait for the parent to signal that we can start
				std::array<char, 128> startBuffer {};
				read(startPipe[readPipe], startBuffer.data(), startBuffer.size());
				close(startPipe[readPipe]);

				// Start the child process
				std::string command = "\"" + path_ + "\" " + Utility::Text::join(parameters_, " ");
				Utility::Logging::logInformation("Starting application process with command %s...", command.c_str());
				std::unique_ptr<FILE, decltype(&pclose)> childPipe(popen(command.c_str(), "r"), [](FILE* file) {
					const auto returnCode = WEXITSTATUS(pclose(file));

					if(returnCode != 0) {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Application threw an error");
					} else {
						return returnCode;
					}
				});
				if(!childPipe) {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not set up child pipe");
				}

				// Receive output
				Utility::Logging::logInformation("Capturing application output...");
				std::array<char, 128> outputBuffer {};
				while(fgets(outputBuffer.data(), outputBuffer.size(), childPipe.get()) != nullptr) {
					auto output = std::string(outputBuffer.data());

					// Send output to parent
					//Utility::Logging::logInformation("Processing application output...\n%s", output.c_str());
					write(outputPipe[writePipe], output.c_str(), output.size());
				}
				close(outputPipe[writePipe]);

				exit(0);
			}
		}

		Application::Application(std::string path, std::vector<std::string> parameters, std::vector<std::shared_ptr<Hardware::CPU>> cpuAffinity, std::shared_ptr<Hardware::GPU> gpu)
			: path_(std::move(path))
			, parameters_(std::move(parameters))
			, cpuAffinity_(std::move(cpuAffinity))
			, gpu_(std::move(gpu)) {
		}

		Application::Application(const Application& application) : Application(application.path_, application.parameters_, application.cpuAffinity_, application.gpu_) {
		}

		std::vector<std::string> Application::getParameters() const {
			return parameters_;
		}

		void Application::setParameters(const std::vector<std::string>& parameters) {
			parameters_ = parameters;
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
			Utility::Logging::logInformation(
				"Changing the applications affinity to %s...",
				Utility::Text::join(
					[&] {
						std::vector<unsigned int> result;
						std::transform(affinity.begin(), affinity.end(), std::back_inserter(result), [](const auto& cpu) {
							return cpu->getID();
						});

						return result;
					}(),
					",")
					.c_str());
			if(sched_setaffinity(processID_, sizeof(mask), &mask) != 0) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not set affinity");
			}
		}

		std::shared_ptr<Hardware::GPU> Application::getGPU() const {
			return gpu_;
		}

		void Application::setGPU(const std::shared_ptr<Hardware::GPU>& gpu) {
			gpu_ = gpu;
		}

		std::string Application::getExecutableOutput() const {
			return executableOutput_;
		}
	}
}