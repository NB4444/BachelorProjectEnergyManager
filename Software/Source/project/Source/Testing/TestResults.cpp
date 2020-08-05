#include "./TestResults.hpp"

#include "Utility/Serialization.hpp"

#include <utility>

namespace Testing {
	std::map<std::string, std::string> TestResults::onSave() {
		return {
			{ "testID", Utility::Serialization::serialize(getTest().getID()) },
			{ "results", Utility::Serialization::serialize(getResults()) }
		};
	}

	TestResults::TestResults(const std::map<std::string, std::string>& row)
		: TestResults(
			Entity<Test>::load(Utility::Serialization::deserializeToInt(row.at("testID"))),
			Utility::Serialization::deserializeToMapOfStringsToStrings(row.at("results"))) {
	}

	TestResults::TestResults(Test test, std::map<std::string, std::string> results)
		: Persistence::Entity<TestResults>("TestResults")
		, test_(std::move(test))
		, results_(std::move(results)) {
	}

	Test TestResults::getTest() const {
		return test_;
	}

	std::map<std::string, std::string> TestResults::getResults() const {
		return results_;
	}
}