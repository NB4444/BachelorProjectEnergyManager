#include "./Application.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/Core.hpp"
#include "EnergyManager/Utility/Environment.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <boost/process.hpp>
#include <boost/process/env.hpp>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sched.h>
#include <stdexcept>
#include <unistd.h>
#include <utility>
#include <wait.h>

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
			// Inject libraries
			std::vector<std::string> librariesToInject = {};
			if(injectReporter_) {
				librariesToInject.push_back(REPORTER_LIBRARY);
			}
			std::string librariesString = Text::join(librariesToInject, ":");

			// Start the child process
			logDebug("Starting application process with path %s, parameters %s and libraries %s...", path_.c_str(), Text::join(parameters_, " ").c_str(), librariesString.c_str());

			// Start the process and capture the output
			executableOutput_ = "";
			boost::process::ipstream output;
			boost::process::ipstream error;
			boost::process::child executable(path_, parameters_, boost::process::std_out > output, boost::process::std_err > error, boost::process::env["LD_PRELOAD"] = librariesString);

			// Monitor output
			std::thread outputMonitor([&] {
				std::string line;
				while(output && std::getline(output, line) && !line.empty()) {
					logInformation("Output: %s", Utility::Text::trim(line).c_str());
					executableOutput_ += line;
				}
			});
			std::thread errorMonitor([&] {
				std::string line;
				while(error && std::getline(error, line) && !line.empty()) {
					logInformation("Error: %s", Utility::Text::trim(line).c_str());
					executableOutput_ += line;
				}
			});

			logDebug("Retrieving child process ID...");

			processID_ = executable.id();

			logDebug("Process ID: %d", processID_);

			// Set the application's affinity
			if(!affinity_.empty()) {
				setAffinity(affinity_);
			}

			// Wait for everything to finish
			logDebug("Waiting for executable to finish...");
			outputMonitor.join();
			errorMonitor.join();
			executable.wait();
		}

		Application::Application(
			std::string path,
			std::vector<std::string> parameters,
			std::vector<std::shared_ptr<Hardware::CentralProcessor>> affinity,
			std::shared_ptr<Hardware::GPU> gpu,
			const bool& logOutput,
			const bool& injectReporter)
			: path_(std::move(path))
			, parameters_(std::move(parameters))
			, affinity_(std::move(affinity))
			, gpu_(std::move(gpu))
			, logOutput_(logOutput)
			, injectReporter_(injectReporter) {
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

		int Application::getGroupID() {
			return groupID_;
		}

		void Application::setGroupID(const int& groupID) {
			groupID_ = groupID;
		}

		int Application::getUserID() {
			return userID_;
		}

		void Application::setUserID(const int& userID) {
			userID_ = userID;
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
			logDebug("Successfully changed affinity");
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