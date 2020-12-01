#include <EnergyManager/Profiling/Profilers/BFSProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUBLASProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUFFTProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/JacobiProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/KMeansProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/MatrixMultiplyProfiler.hpp>
#include <EnergyManager/Utility/Persistence/Entity.hpp>
#include <EnergyManager/Utility/Text.hpp>

int main(int argumentCount, char *argumentValues[]) {
  // Parse arguments
  static const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(
      argumentCount, argumentValues);

  // Load the database
  EnergyManager::Utility::Persistence::Entity::initialize(
      EnergyManager::Utility::Text::getArgument<std::string>(
          arguments, "--database", std::string(PROJECT_DATABASE)));

  // Run the profilers
  EnergyManager::Profiling::Profilers::BFSProfiler(arguments).run();
  EnergyManager::Profiling::Profilers::CUBLASProfiler(arguments).run();
  EnergyManager::Profiling::Profilers::CUFFTProfiler(arguments).run();
  EnergyManager::Profiling::Profilers::JacobiProfiler(arguments).run();
  EnergyManager::Profiling::Profilers::KMeansProfiler(arguments).run();
  EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments).run();
}