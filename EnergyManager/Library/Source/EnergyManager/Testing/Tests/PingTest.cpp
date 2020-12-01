#include "./PingTest.hpp"

#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			PingTest::PingTest(
				const std::string& name,
				const std::string& host,
				const unsigned int& times,
				const std::chrono::system_clock::duration& applicationMonitorPollingInterval,
				const std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>>& monitors)
				: ApplicationTest(
					name,
					EnergyManager::Utility::Application("/bin/ping", { "-c " + std::to_string(times), host }, {}, nullptr),
					{
						{ "Packets Transmitted", "(\\d+) packets transmitted" },
						{ "Packets Received", "(\\d+) received" },
						{ "Packets Lost", "(\\d+)% packet loss" },
						{ "Time", "time (\\d+)" },
					},
					applicationMonitorPollingInterval,
					monitors) {
			}

			PingTest::PingTest(const std::map<std::string, std::string>& arguments)
				: PingTest(
					"PingTest",
					Utility::Text::getArgument<std::string>(arguments, "--host", "8.8.8.8"),
					Utility::Text::getArgument<unsigned int>(arguments, "--times", 5),
					Utility::Text::getArgument<std::chrono::system_clock::duration>(
						arguments,
						"--applicationMonitorInterval",
						Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--monitorInterval", std::chrono::milliseconds(100)))) {
			}
		}
	}
}