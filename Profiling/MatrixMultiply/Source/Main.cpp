#include <EnergyManager/Profiling/Profilers/MatrixMultiplyProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}