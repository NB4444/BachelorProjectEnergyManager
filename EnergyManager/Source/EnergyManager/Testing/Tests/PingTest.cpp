#include "./PingTest.hpp"

#include "EnergyManager/Utility/Exceptions/ParseException.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			void PingTest::initialize() {
				Test::addParser([](const std::string& name,
								   const std::map<std::string, std::string>& parameters,
								   const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors) {
					if(name != "PingTest") {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_PARSE_EXCEPTION();
					}

					return std::make_shared<EnergyManager::Testing::Tests::PingTest>(
						Utility::Text::getParameter(parameters, "name"),
						Utility::Text::getParameter(parameters, "host"),
						std::stoi(Utility::Text::getParameter(parameters, "times")),
						std::chrono::duration_cast<std::chrono::system_clock::duration>(
							std::chrono::milliseconds(std::stoul(Utility::Text::getParameter(parameters, "applicationMonitorPollingInterval")))),
						monitors);
				});
			}

			PingTest::PingTest(
				const std::string& name,
				const std::string& host,
				const unsigned int& times,
				std::chrono::system_clock::duration applicationMonitorPollingInterval,
				std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors)
				: ApplicationTest(
					name,
					Application("/bin/ping"),
					{ "-c " + std::to_string(times), host },
					{},
					nullptr,
					{
						{ "Packets Transmitted", "(\\d+) packets transmitted" },
						{ "Packets Received", "(\\d+) received" },
						{ "Packets Lost", "(\\d+)% packet loss" },
						{ "Time", "time (\\d+)" },
					},
					applicationMonitorPollingInterval,
					monitors) {
			}
		}
	}
}