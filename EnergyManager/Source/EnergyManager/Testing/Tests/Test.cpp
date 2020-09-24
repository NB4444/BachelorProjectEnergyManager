#include "./Test.hpp"

#include "EnergyManager/Testing/TestResults.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <mutex>
#include <regex>
#include <thread>
#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::vector<Test::Parser> Test::parsers_ = {};

			void Test::addParser(const Test::Parser& parser) {
				parsers_.push_back(parser);
			}

			std::map<std::string, std::string> Test::onRun() {
				return {};
			}

			void Test::onSave() {
				// Insert the Test
				setID(insert("Tests", { { "name", '"' + getName() + '"' } }));
			}

			std::shared_ptr<Test> Test::parse(
				const std::string& name,
				const std::map<std::string, std::string>& parameters,
				const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors) {
				for(const auto& parser : parsers_) {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(return parser(name, parameters, monitors));
				}

				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not parse Test");
			}

			Test::Test(std::string name, std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors) : name_(std::move(name)), monitors_(std::move(monitors)) {
				// Create the table if it does not exist yet
				try {
					createTable("Tests", { { "id", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL" }, { "name", "TEXT NOT NULL" } });
				} catch(const std::runtime_error& error) {
				}
			}

			std::string Test::getName() const {
				return name_;
			}

			TestResults Test::run() {
				// Start the Monitors
				std::map<std::shared_ptr<Monitoring::Monitor>, std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>> monitorResults;
				std::mutex monitorMutex;
				std::map<std::shared_ptr<Monitoring::Monitor>, std::thread> monitors;
				for(const auto& monitor : monitors_) {
					monitors[monitor.first] = std::thread([&] {
						monitor.first->run(monitor.second);

						// Store the Monitor results
						std::lock_guard<std::mutex> guard(monitorMutex);
						monitorResults[monitor.first] = monitor.first->getVariableValues();
					});
				}

				// Run the Test
				std::map<std::string, std::string> results = onRun();

				// Collect final results
				for(auto& monitor : monitors) {
					monitor.first->stop();
				}

				// Wait for monitors to exit
				for(auto& monitor : monitors) {
					monitor.second.join();
				}

				return TestResults(*this, results, monitorResults);
			}
		}
	}
}