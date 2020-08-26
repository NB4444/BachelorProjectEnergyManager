#include "./Test.hpp"

#include "EnergyManager/Testing/TestResults.hpp"

#include <iostream>
#include <mutex>
#include <regex>
#include <thread>
#include <utility>
#include <vector>
#include <zconf.h>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> Test::onRun() {
				return {};
			}

			Test::Test(std::string name, std::map<std::shared_ptr<Profiling::Monitor>, std::chrono::system_clock::duration> monitors) : name_(std::move(name)), monitors_(std::move(monitors)) {
			}

			std::string Test::getName() const {
				return name_;
			}

			TestResults Test::run(const std::string& databaseFile) {
				// Start the Monitors
				std::map<std::shared_ptr<Profiling::Monitor>, std::map<std::chrono::system_clock::time_point, std::map<std::string, std::string>>> monitorResults;
				std::mutex monitorResultsMutex;
				std::map<std::shared_ptr<Profiling::Monitor>, std::thread> monitors;
				for(const auto& monitor : monitors_) {
					monitors[monitor.first] = std::thread([&] {
						monitor.first->run(monitor.second);

						// Store the Monitor results
						std::lock_guard<std::mutex> guard(monitorResultsMutex);
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

				return TestResults(databaseFile, *this, results, monitorResults);
			}
		}
	}
}