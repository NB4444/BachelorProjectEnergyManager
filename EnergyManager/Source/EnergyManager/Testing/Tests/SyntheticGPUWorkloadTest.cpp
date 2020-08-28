#include "./SyntheticGPUWorkloadTest.hpp"

#include "EnergyManager/Profiling/CPUMonitor.hpp"
#include "EnergyManager/Profiling/GPUMonitor.hpp"
#include "EnergyManager/Profiling/NodeMonitor.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> SyntheticGPUWorkloadTest::onRun() {
				workload_->run();

				return {};
			}

			SyntheticGPUWorkloadTest::SyntheticGPUWorkloadTest(
				const std::string& name,
				std::shared_ptr<Benchmarking::SyntheticGPUWorkload> workload,
				const std::shared_ptr<Hardware::Node>& node,
				const std::shared_ptr<Hardware::CPU>& cpu,
				const std::shared_ptr<Hardware::GPU>& gpu)
				: Test(
					name,
					{ { std::shared_ptr<Profiling::Monitor>(new Profiling::GPUMonitor(gpu)), std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(100)) },
					  { std::shared_ptr<Profiling::Monitor>(new Profiling::CPUMonitor(cpu)), std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(100)) },
					  { std::shared_ptr<Profiling::Monitor>(new Profiling::NodeMonitor(node)), std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(100)) } })
				, workload_(std::move(workload)) {
			}
		}
	}
}