#include <EnergyManager/Profiling/Profilers/JacobiProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Profiling::Profilers::JacobiProfiler(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}