#include "Application.hpp"
#include "Test.hpp"
#include "TestResults.hpp"

int main() {
	// Set up a database to handle Test results
	TestResults testResults(TEST_RESULTS_DATABASE);

	// Run a Test and save the results
	Test test(Application("/bin/ping"));
	testResults.insert(test.execute({ "-c 4", "google.com" }, {}));

	return 0;
}
