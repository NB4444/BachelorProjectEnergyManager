#include <EnergyManager/Testing/Tests/ActiveInactiveWorkloadTest.hpp>
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

  // Run the profiler
  EnergyManager::Testing::Tests::ActiveInactiveWorkloadTest(arguments).run();
}