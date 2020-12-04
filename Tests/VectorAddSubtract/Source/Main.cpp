#include <EnergyManager/Testing/Tests/VectorAddSubtractTest.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Testing::Tests::VectorAddSubtractTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}
