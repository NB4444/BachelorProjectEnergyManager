#include "./PingTest.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			PingTest::PingTest(const std::string& name, const std::string& host, const unsigned int& times)
				: ApplicationTest(
					name,
					Application("/bin/ping"),
					{ "-c " + std::to_string(times), host },
					{
						{ "Packets Transmitted", "(\\d+) packets transmitted" },
						{ "Packets Received", "(\\d+) received" },
						{ "Packets Lost", "(\\d+)% packet loss" },
						{ "Time", "time (\\d+)" },
					}) {
			}
		}
	}
}