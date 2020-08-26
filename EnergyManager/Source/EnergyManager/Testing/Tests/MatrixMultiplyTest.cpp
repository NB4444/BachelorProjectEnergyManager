#include "./MatrixMultiplyTest.hpp"

#include "EnergyManager/Profiling/CPUMonitor.hpp"
#include "EnergyManager/Profiling/GPUMonitor.hpp"
#include "EnergyManager/Profiling/NodeMonitor.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#include <chrono>
#include <stdexcept>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			MatrixMultiplyTest::MatrixMultiplyTest(
				const std::string& name,
				const std::shared_ptr<Hardware::CPU>& cpu,
				const std::shared_ptr<Hardware::GPU>& gpu,
				const size_t& matrixAWidth,
				const size_t& matrixAHeight,
				const size_t& matrixBWidth,
				const size_t& matrixBHeight)
				: ApplicationTest(
				name,
				Application("/usr/local/cuda-10.1/samples/0_Simple/matrixMul/matrixMul"),
				{
					"-device=" + std::to_string(gpu->getID()),
					"-wA=" + std::to_string(matrixAWidth),
					"-wB=" + std::to_string(matrixBWidth),
					"-hA=" + std::to_string(matrixAHeight),
					"-hB=" + std::to_string(matrixBHeight)
				},
				{
					{ "performance", "Performance= (.+?)," },
					{ "time", "Time= (.+?)," },
					{ "size", "Size= (.+?)," },
					{ "workgroupSize", "WorkgroupSize= (.+?)\n" },
				},
				{
					{ std::shared_ptr<Profiling::Monitor>(new Profiling::GPUMonitor(gpu)), std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(100)) },
					{ std::shared_ptr<Profiling::Monitor>(new Profiling::CPUMonitor(cpu)), std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(100)) },
					{ std::shared_ptr<Profiling::Monitor>(new Profiling::NodeMonitor(cpu, gpu)), std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(100)) }
				}) {
				if(matrixAWidth % 32 != 0 || matrixBWidth % 32 != 0 || matrixAHeight % 32 != 0 || matrixBHeight % 32 != 0) {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Matrix dimensions must be a multiple of 32");
				}
			}
		}
	}
}