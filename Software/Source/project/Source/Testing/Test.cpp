#include "./Test.hpp"

#include "Testing/TestResults.hpp"
#include "Utility/Serialization.hpp"

#include <iostream>
#include <regex>
#include <utility>

namespace Testing {
	std::map<std::string, std::string> Test::onSave() {
		return {
			{ "name", getName() },
			{ "applicationID", Utility::Serialization::serialize(getApplication().getID()) },
			{ "parameters", Utility::Serialization::serialize(getParameters()) },
			{ "results", Utility::Serialization::serialize(getResults()) }
		};
	}

	Test::Test(const std::map<std::string, std::string>& row)
		: Test(
			row.at("name"),
			Entity<Application>::load(Utility::Serialization::deserializeToInt(row.at("applicationID"))),
			Utility::Serialization::deserializeToVectorOfStrings(row.at("parameters")),
			Utility::Serialization::deserializeToMapOfStringsToStrings(row.at("results"))) {
	}

	Test::Test(std::string name, const Application& application, std::vector<std::string> parameters, std::map<std::string, std::string> results)
		: Persistence::Entity<Test>("Test")
		, name_(std::move(name))
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

	std::map<std::string, std::string> Test::getResults() const {
		return results_;
	}

	TestResults Test::run() {
		// Run the Application
		application_.start(parameters_);
		application_.waitUntilDone();

		// Get output
		std::string output = application_.getExecutableOutput();

		// Parse results
		std::map<std::string, std::string> results;
		for(const auto& result : results_) {
			std::smatch match;
			std::regex regex(result.second);
			if(std::regex_search(output, match, regex)) {
				results[result.first] = match.str(1);
			}
		}

		return TestResults(*this, results);
	}
}