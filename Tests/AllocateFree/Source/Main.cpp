#include <EnergyManager/Testing/Tests/AllocateFreeWorkloadTest.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Testing::Tests::AllocateFreeWorkloadTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}
