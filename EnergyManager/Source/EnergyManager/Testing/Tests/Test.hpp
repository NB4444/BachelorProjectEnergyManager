#pragma once

#include "EnergyManager/Monitoring/Monitor.hpp"
#include "EnergyManager/Persistence/Entity.hpp"
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
			class Test : public Persistence::Entity {
				using Parser = std::function<std::shared_ptr<Test>(
					const std::string& name,
					const std::map<std::string, std::string>& parameters,
					const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors)>;

				static std::vector<Parser> parsers_;

				/**
				 * The name of the Test.
				 */
				std::string name_;

				/**
				 * The monitors to run during the Test.
				 */
				std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors_;

			protected:
				static void addParser(const Parser& parser);

				/**
				 * Executes the Test.
				 * @return The results of the Test.
				 */
				virtual std::map<std::string, std::string> onRun();

				void onSave() override;

			public:
				static std::shared_ptr<Test> parse(
					const std::string& name,
					const std::map<std::string, std::string>& parameters,
					const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors);

				/**
				 * Creates a new Test.
				 * @param name The name of the Test.
				 * @param monitors The monitors to run during the Test and their associated polling intervals.
				 */
				Test(std::string name, std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors = {});

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