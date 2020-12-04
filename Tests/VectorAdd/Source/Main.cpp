#include <EnergyManager/Testing/Tests/VectorAddWorkloadTest.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Testing::Tests::VectorAddWorkloadTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}
