#include "Test.hpp"

#include <iostream>
#include <utility>

namespace Testing {
	Test::Test(std::string name, const Application& application, std::vector<std::string> parameters, std::map<std::string, std::regex> results)
		: name_(std::move(name))
		, application_(application)
		, parameters_(std::move(parameters))
		, results_(std::move(results)) {
	}

	std::string Test::getName() const {
		return name_;
	}

	Application Test::getApplication() const {
		return application_;
	}

	std::vector<std::string> Test::getParameters() const {
		return parameters_;
	}

	std::map<std::string, std::regex> Test::getResults() const {
		return results_;
	}

	std::map<std::string, std::string> Test::run() {
		application_.start(parameters_);

		application_.waitUntilDone();

		std::cout << "[EXECUTABLE]\n"
				  << application_.getExecutableOutput() << std::endl;
		std::cout << "[CUDA ENERGY MONITOR]\n"
				  << application_.getCUDAEnergyMonitorOutput() << std::endl;

		// TODO: Parse and return results

		return {};
	}
}