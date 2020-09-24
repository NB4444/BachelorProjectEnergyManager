#include "./MatrixMultiplyTest.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Exceptions/ParseException.hpp"

#include <chrono>
#include <stdexcept>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			void MatrixMultiplyTest::initialize() {
				Test::addParser([](const std::string& name,
								   const std::map<std::string, std::string>& parameters,
								   const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors) {
					if(name != "MatrixMultiplyTest") {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_PARSE_EXCEPTION();
					}

					return std::make_shared<EnergyManager::Testing::Tests::MatrixMultiplyTest>(
						Utility::Text::getParameter(parameters, "name"),
						Hardware::CPU::parseCPUs(Utility::Text::getParameter(parameters, "cpu")),
						EnergyManager::Hardware::GPU::getGPU(std::stoi(Utility::Text::getParameter(parameters, "gpu"))),
						std::stoi(Utility::Text::getParameter(parameters, "matrixAWidth")),
						std::stoi(Utility::Text::getParameter(parameters, "matrixAHeight")),
						std::stoi(Utility::Text::getParameter(parameters, "matrixBWidth")),
						std::stoi(Utility::Text::getParameter(parameters, "matrixBHeight")),
						std::chrono::duration_cast<std::chrono::system_clock::duration>(
							std::chrono::milliseconds(std::stoul(Utility::Text::getParameter(parameters, "applicationMonitorPollingInterval")))),
						monitors);
				});
			}

			MatrixMultiplyTest::MatrixMultiplyTest(
				const std::string& name,
				const std::vector<std::shared_ptr<Hardware::CPU>>& cpus,
				const std::shared_ptr<Hardware::GPU>& gpu,
				const size_t& matrixAWidth,
				const size_t& matrixAHeight,
				const size_t& matrixBWidth,
				const size_t& matrixBHeight,
				std::chrono::system_clock::duration applicationMonitorPollingInterval,
				const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors)
				: ApplicationTest(
					name,
					Application(std::string(PROJECT_RESOURCES_DIRECTORY) + "/CUDA/Samples/0_Simple/matrixMul/matrixMul"),
					{ "-device=" + std::to_string(gpu->getID()),
					  "-wA=" + std::to_string(matrixAWidth),
					  "-wB=" + std::to_string(matrixBWidth),
					  "-hA=" + std::to_string(matrixAHeight),
					  "-hB=" + std::to_string(matrixBHeight) },
					cpus,
					gpu,
					{
						{ "performance", "Performance= (.+?)," },
						{ "time", "Time= (.+?)," },
						{ "size", "Size= (.+?)," },
						{ "workgroupSize", "WorkgroupSize= (.+?)\n" },
					},
					applicationMonitorPollingInterval,
					monitors) {
				if(matrixAWidth % 32 != 0 || matrixBWidth % 32 != 0 || matrixAHeight % 32 != 0 || matrixBHeight % 32 != 0) {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Matrix dimensions must be a multiple of 32");
				}
			}
		}
	}
}