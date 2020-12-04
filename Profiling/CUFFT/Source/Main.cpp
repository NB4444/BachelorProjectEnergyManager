#include <EnergyManager/Profiling/Profilers/CUFFTProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	EnergyManager::Profiling::Profilers::CUFFTProfiler(EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues)).run();
}