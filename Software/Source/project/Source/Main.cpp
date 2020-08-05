#include "Application.hpp"
#include "Testing/Test.hpp"
#include "Testing/TestResults.hpp"
#include "Testing/TestRunner.hpp"

int main() {
	// Set up a new TestRunner
	Testing::TestRunner testRunner;

	// Add some tests
	testRunner.addTest(Testing::Test("Ping", Application("/bin/ping"), { "-c 4", "google.com" }, {
																									 { "Packets Transmitted", "(\\d+) packets transmitted" },
																									 { "Packets Received", "(\\d+) received" },
																									 { "Packets Lost", "(\\d+)% packet loss" },
																									 { "Time", "time (\\d+)" },
																								 }));

	// Run the tests
	testRunner.run();

	return 0;
}