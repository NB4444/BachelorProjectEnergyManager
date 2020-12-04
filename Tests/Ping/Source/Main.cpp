#include <EnergyManager/Testing/Tests/PingTest.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Testing::Tests::PingTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}
