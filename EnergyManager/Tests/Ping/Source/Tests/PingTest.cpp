#include "./PingTest.hpp"

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
}
