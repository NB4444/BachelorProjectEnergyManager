#include "./Application.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <iostream>
#include <memory>
#include <sched.h>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace EnergyManager {
	namespace Utility {
		std::vector<std::string> Application::generateHeaders() const {
			auto headers = Runnable::generateHeaders();
			headers.push_back("Application " + getPath());

			return headers;
		}

		void Application::beforeRun() {
			// Set the active GPU
			if(gpu_ != nullptr) {
				logDebug("Changing active GPU to %d...", gpu_->getID());
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
			logTrace("Setting up pipe to application...");
			int outputPipe[2]; // Pipe to transfer output from child to parent
			if(pipe(outputPipe) == -1) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not set up output pipe");
			}
			int startPipe[2]; // Pipe to send start signal from parent to child
			if(pipe(startPipe) == -1) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not set up start pipe");
			}

			// Fork the current process and open a set of pipes
			logTrace("Forking current process...");

			// Flush all output buffers to prevent duplicate output caused by a double-flush
			// This happens because the fork duplicates all output buffers
			Utility::Logging::flush();

			if((processID_ = fork()) < 0) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not fork current process");
			}
			bool isParent = processID_ > 0;

			// Set up the pipes
			if(isParent) {
				logTrace("Configuring pipe in parent process...");
				close(outputPipe[writePipe]);
				close(startPipe[readPipe]);

				// Set the application's affinity
				if(!affinity_.empty()) {
					setAffinity(affinity_);
				}

				// Signal on the start pipe
				std::string signal = "START";
				write(startPipe[writePipe], signal.c_str(), signal.size());
				close(startPipe[writePipe]);

				// Get a pointer to capture application output
				logTrace("Capturing child output...");
				FILE* filePointer = fdopen(outputPipe[readPipe], "r");
				std::array<char, BUFSIZ> outputBuffer {};
				while(fgets(outputBuffer.data(), outputBuffer.size(), filePointer) != nullptr) {
					auto output = std::string(outputBuffer.data());

					if(logOutput_) {
						logInformation("Output: %s", Utility::Text::trim(output).c_str());
					}
					executableOutput_ += output;
				}
				close(outputPipe[readPipe]);

				// Clean up
				fclose(filePointer);
			} else {
				logTrace("Configuring pipe in child process...");
				close(outputPipe[readPipe]);
				close(startPipe[writePipe]);

				// Wait for the parent to signal that we can start
				std::array<char, 128> startBuffer {};
				read(startPipe[readPipe], startBuffer.data(), startBuffer.size());
				close(startPipe[readPipe]);

				//// Start the child process
				//std::string command = "\"" + path_ + "\"";
				//if(!parameters_.empty()) {
				//	command += " \"" + Utility::Text::join(parameters_, "\" \"") + "\"";
				//}
				//command += " 2>&1";

				// Flush the buffers
				Utility::Logging::flush();

				// Redirect the output
				dup2(outputPipe[writePipe], STDOUT_FILENO);
				dup2(outputPipe[writePipe], STDERR_FILENO);

				// Prepare parameters
				parameters_.insert(parameters_.begin(), path_);
				std::vector<char*> cParameters;
				cParameters.reserve(parameters_.size() + 1);
				std::transform(parameters_.begin(), parameters_.end(), std::back_inserter(cParameters), [](std::string& string) {
					return const_cast<char*>(string.data());
				});
				cParameters.push_back(nullptr);

				// Start the application
				logTrace("Starting application process with path %s and parameters %s...", path_.c_str(), Utility::Text::join(parameters_, " ").c_str());
				execv(path_.c_str(), cParameters.data());

				//std::unique_ptr<FILE, decltype(&pclose)> childPipe(popen(command.c_str(), "r"), [](FILE* file) {
				//	const auto rawReturnCode = pclose(file);
				//	const auto returnCode = WEXITSTATUS(rawReturnCode);
				//
				//	if(returnCode != 0) {
				//		ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Application threw an error");
				//	} else {
				//		return returnCode;
				//	}
				//});
				//if(!childPipe) {
				//	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not set up child pipe");
				//}
				//
				//// Receive output
				//logDebug("Capturing application output...");
				//std::array<char, 128> outputBuffer {};
				//while(fgets(outputBuffer.data(), outputBuffer.size(), childPipe.get()) != nullptr) {
				//	auto output = std::string(outputBuffer.data());
				//
				//	// Send output to parent
				//	//logDebug("Processing application output: %s", Utility::Text::trim(output).c_str());
				//	write(outputPipe[writePipe], output.c_str(), output.size());
				//}
				close(outputPipe[writePipe]);

				exit(0);
			}
		}

		Application::Application(
			std::string path,
			std::vector<std::string> parameters,
			std::vector<std::shared_ptr<Hardware::CentralProcessor>> affinity,
			std::shared_ptr<Hardware::GPU> gpu,
			const bool& logOutput)
			: path_(std::move(path))
			, parameters_(std::move(parameters))
			, affinity_(std::move(affinity))
			, gpu_(std::move(gpu))
			, logOutput_(logOutput) {
		}

		Application::Application(const Application& application) : Application(application.path_, application.parameters_, application.affinity_, application.gpu_) {
		}

		std::string Application::getPath() const {
			return path_;
		}

		void Application::setPath(const std::string& path) {
			path_ = path;
		}

		std::vector<std::string> Application::getParameters() const {
			return parameters_;
		}

		void Application::setParameters(const std::vector<std::string>& parameters) {
			parameters_ = parameters;
		}

		std::vector<std::shared_ptr<Hardware::CentralProcessor>> Application::getAffinity() const {
			// Get the affinity mask
			cpu_set_t cpuMask;
			if(sched_getaffinity(processID_, sizeof(cpuMask), &cpuMask) != 0) {
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not get affinity");
			}

			// Decode the mask
			std::vector<std::shared_ptr<Hardware::CentralProcessor>> affinity;
			for(const auto& cpu : Hardware::CPU::getCPUs()) {
				for(const auto& core : cpu->getCores()) {
					if(CPU_ISSET(core->getCoreID(), &cpuMask)) {
						affinity.push_back(cpu);
					}
				}
			}

			return affinity;
		}

		void Application::setAffinity(const std::vector<std::shared_ptr<Hardware::CentralProcessor>>& affinity) {
			std::vector<std::shared_ptr<Hardware::CentralProcessor>> processors;
			for(const auto& processor : affinity) {
				if(dynamic_cast<Hardware::CPU*>(processor.get())) {
					for(const auto& core : std::dynamic_pointer_cast<Hardware::CPU>(processor)->getCores()) {
						processors.push_back(core);
					}
				} else {
					processors.push_back(processor);
				}
			}

			// Encode an affinity mask
			cpu_set_t mask;
			CPU_ZERO(&mask);
			for(const auto& processor : processors) {
				CPU_SET(processor->getID(), &mask);
			}

			// Set the affinity
			logDebug(
				"Changing the application's affinity to %s...",
				Utility::Text::join(
					[&] {
						std::vector<unsigned int> result;
						std::transform(processors.begin(), processors.end(), std::back_inserter(result), [](const auto& processor) {
							return processor->getID();
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

		bool Application::getLogOutput() const {
			return logOutput_;
		}

		void Application::setLogOutput(const bool& logOutput) {
			logOutput_ = logOutput;
		}

		std::string Application::getExecutableOutput() const {
			return executableOutput_;
		}
	}
}