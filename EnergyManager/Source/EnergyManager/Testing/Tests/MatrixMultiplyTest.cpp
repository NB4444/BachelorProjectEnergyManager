#include "./MatrixMultiplyTest.hpp"

#include "EnergyManager/Monitoring/CPUMonitor.hpp"
#include "EnergyManager/Monitoring/GPUMonitor.hpp"
#include "EnergyManager/Monitoring/NodeMonitor.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <chrono>
#include <stdexcept>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			MatrixMultiplyTest::MatrixMultiplyTest(
				const std::string& name,
				const std::shared_ptr<Hardware::Node>& node,
				const std::shared_ptr<Hardware::CPU>& cpu,
				const std::shared_ptr<Hardware::GPU>& gpu,
				const size_t& matrixAWidth,
				const size_t& matrixAHeight,
				const size_t& matrixBWidth,
				const size_t& matrixBHeight,
				std::chrono::system_clock::duration applicationMonitorPollingInterval,
				std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors)
				: ApplicationTest(
					name,
					Application(std::string(PROJECT_RESOURCES_DIRECTORY) + "/CUDA/Samples/0_Simple/matrixMul/matrixMul"),
					{ "-device=" + std::to_string(gpu->getID()),
					  "-wA=" + std::to_string(matrixAWidth),
					  "-wB=" + std::to_string(matrixBWidth),
					  "-hA=" + std::to_string(matrixAHeight),
					  "-hB=" + std::to_string(matrixBHeight) },
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