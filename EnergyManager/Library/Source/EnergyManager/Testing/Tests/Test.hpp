#pragma once

#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"
#include "EnergyManager/Monitoring/Profilers/Profiler.hpp"
#include "EnergyManager/Testing/Application.hpp"

#include <map>
#include <memory>
#include <string>

namespace EnergyManager {
	namespace Testing {
		class TestResults;

		namespace Tests {
			/**
			 * A test of an Application.
			 */
			class Test : public Monitoring::Profilers::Profiler {
				/**
				 * The name of the Test.
				 */
				std::string name_;

				/**
				 * The test results.
				 */
				std::map<std::string, std::string> testResults_;

			protected:
				void onProfile(const std::map<std::string, std::string>& profile) final;

				/**
				 * Executes the Test.
				 * @return The results of the Test.
				 */
				virtual std::map<std::string, std::string> onTest();

			public:
				/**
				 * Creates a new Test.
				 * @param name The name of the Test.
				 * @param monitors The monitors to run during the Test.
				 */
				explicit Test(std::string name, const std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>>& monitors);

				/**
				 * Gets the name of the Test.
				 * @return The name.
				 */
				std::string getName() const;

				/**
				 * Gets the results of the latest execution.
				 * @return The results.
				 */
				std::map<std::string, std::string> getTestResults() const;
			};
		}
	}
}