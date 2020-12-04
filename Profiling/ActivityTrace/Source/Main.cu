#include <EnergyManager/Profiling/Profilers/ActivityTraceProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Profiling::Profilers::ActivityTraceProfiler(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}
