#include <EnergyManager/Testing/Tests/ActiveInactiveWorkloadTest.hpp>
#include <EnergyManager/Testing/Tests/AllocateFreeWorkloadTest.hpp>
#include <EnergyManager/Testing/Tests/PingTest.hpp>
#include <EnergyManager/Testing/Tests/VectorAddSubtractTest.hpp>
#include <EnergyManager/Testing/Tests/VectorAddWorkloadTest.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Testing::Tests::ActiveInactiveWorkloadTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
	EnergyManager::Testing::Tests::AllocateFreeWorkloadTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
	//EnergyManager::Testing::Tests::PingTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
	EnergyManager::Testing::Tests::VectorAddWorkloadTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
	EnergyManager::Testing::Tests::VectorAddSubtractTest(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}