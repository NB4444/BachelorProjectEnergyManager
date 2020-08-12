#pragma once

#include "EnergyManager/Application.hpp"
#include "EnergyManager/Profiling/Monitor.hpp"

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
			class Test {
				/**
				 * The name of the Test.
				 */
				std::string name_;

				/**
				 * The monitors to run during the Test.
				 */
				std::map<std::shared_ptr<Profiling::Monitor>, std::chrono::seconds> monitors_;

			protected:
				/**
				 * Executes the Test.
				 * @return The results of the Test.
				 */
				virtual std::map<std::string, std::string> onRun();

			public:
				/**
				 * Creates a new Test.
				 * @param name The name of the Test.
				 * @param monitors The monitors to run during the Test and their associated polling intervals.
				 */
				Test(std::string name, std::map<std::shared_ptr<Profiling::Monitor>, std::chrono::seconds> monitors = {});

				/**
				 * Gets the name of the Test.
				 * @return The name.
				 */
				std::string getName() const;

				/**
				 * Runs the Test.
				 * @return The parsed Test results.
				 */
				TestResults run();
			};
		}
	}
}