#include "./PingTest.hpp"

namespace Testing {
	namespace Tests {
		PingTest::PingTest(const std::string& host, const int& times)
			: ApplicationTest("PingTest", Application("/bin/ping"), { "-c " + std::to_string(times), host }, {
																												 { "Packets Transmitted", "(\\d+) packets transmitted" },
																												 { "Packets Received", "(\\d+) received" },
																												 { "Packets Lost", "(\\d+)% packet loss" },
																												 { "Time", "time (\\d+)" },
																											 }) {
		}
	}
}