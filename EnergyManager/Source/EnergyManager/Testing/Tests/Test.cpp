#include "./Test.hpp"

#include "EnergyManager/Testing/TestResults.hpp"
#include "EnergyManager/Utility/Serialization.hpp"

#include <iostream>
#include <regex>
#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> Test::onSave() {
				return {
					{ "name", getName() }
				};
			}

			TestResults Test::onRun() {
				return TestResults(*this, {});
			}

			Test::Test(const std::map<std::string, std::string>& row)
				: Test(row.at("name")) {
			}

			Test::Test(std::string name)
				: Persistence::Entity<Test>("Test")
				, name_(std::move(name)) {
			}

			std::string Test::getName() const {
				return name_;
			}

			TestResults Test::run() {
				return onRun();
			}
		}
	}
}