#include <EnergyManager/Profiling/Profilers/KMeansProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Profiling::Profilers::KMeansProfiler(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}