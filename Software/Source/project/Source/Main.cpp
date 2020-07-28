#include "Application.hpp"

#include <iostream>

int main() {
	Application test("/bin/ping");

	std::cout << test.start({ "-c 4", "google.com" }) << std::endl;

	return 0;
}
