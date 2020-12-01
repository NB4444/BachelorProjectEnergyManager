#pragma once

#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"
#include "EnergyManager/Profiling/Profilers/Profiler.hpp"
#include "EnergyManager/Utility/Application.hpp"

#include <map>
#include <memory>
#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			/**
			 * A test of an Application.
			 */
			class Test : public Profiling::Profilers::Profiler {
				/**
				 * The name of the Test.
				 */
				std::string name_;

				/**
				 * The test results.
				 */
				std::map<std::string, std::string> testResults_;

			protected:
				std::vector<std::string> generateHeaders() const override;

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
				 * Creates a new Test.
				 * @param name The name of the Test.
				 * @param arguments The command line arguments.
				 */
				explicit Test(const std::string& name, const std::map<std::string, std::string>& arguments);

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