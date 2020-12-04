#include <EnergyManager/Profiling/Profilers/BFSProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUBLASProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUFFTProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/JacobiProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/KMeansProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/MatrixMultiplyProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	static const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	// Run the profilers
	EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments).run();
	EnergyManager::Profiling::Profilers::KMeansProfiler(arguments).run();
	EnergyManager::Profiling::Profilers::BFSProfiler(arguments).run();
	EnergyManager::Profiling::Profilers::CUBLASProfiler(arguments).run();
	EnergyManager::Profiling::Profilers::CUFFTProfiler(arguments).run();
	EnergyManager::Profiling::Profilers::JacobiProfiler(arguments).run();
}