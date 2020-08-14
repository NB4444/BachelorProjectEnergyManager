#include "./MatrixMultiplyTest.hpp"

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Profiling/CPUMonitor.hpp"
#include "EnergyManager/Profiling/GPUMonitor.hpp"
#include "EnergyManager/Utility/Exception.hpp"

#include <chrono>
#include <stdexcept>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			MatrixMultiplyTest::MatrixMultiplyTest(
				const std::string& name,
				const Hardware::CPU& cpu,
				const Hardware::GPU& gpu,
				const size_t& matrixAWidth,
				const size_t& matrixAHeight,
				const size_t& matrixBWidth,
				const size_t& matrixBHeight)
				: ApplicationTest(
					name,
					Application("/usr/local/cuda-10.1/samples/0_Simple/matrixMul/matrixMul"),
					{ "-device=" + std::to_string(gpu.getID()),
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
					{ { std::shared_ptr<Profiling::Monitor>(new Profiling::GPUMonitor(gpu)), std::chrono::seconds(1) },
					  { std::shared_ptr<Profiling::Monitor>(new Profiling::CPUMonitor(cpu)), std::chrono::seconds(1) } }) {
				if(matrixAWidth % 32 != 0 || matrixBWidth % 32 != 0 || matrixAHeight % 32 != 0 || matrixBHeight % 32 != 0) {
					ENERGY_MANAGER_UTILITY_EXCEPTION("Matrix dimensions must be a multiple of 32");
				}
			}
		}
	}
}