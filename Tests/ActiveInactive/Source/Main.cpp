#include <EnergyManager/Testing/Tests/ActiveInactiveWorkloadTest.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Testing::Tests::ActiveInactiveWorkloadTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}