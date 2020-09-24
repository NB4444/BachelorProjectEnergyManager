#include "./SyntheticWorkloadTest.hpp"

#include "EnergyManager/Utility/Exceptions/ParseException.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> SyntheticWorkloadTest::onRun() {
				workload_->run(gpu_);

				return {};
			}

			void SyntheticWorkloadTest::initialize() {
				Test::addParser([](const std::string& name,
								   const std::map<std::string, std::string>& parameters,
								   const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors) {
					if(name != "SyntheticWorkloadTest") {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_PARSE_EXCEPTION();
					}

					return std::make_shared<EnergyManager::Testing::Tests::SyntheticWorkloadTest>(
						Utility::Text::getParameter(parameters, "name"),
						Benchmarking::Workloads::SyntheticWorkload::parse(Utility::Text::getParameter(parameters, "workload"), parameters),
						EnergyManager::Hardware::GPU::getGPU(std::stoi(Utility::Text::getParameter(parameters, "gpu"))),
						monitors);
				});
			}

			SyntheticWorkloadTest::SyntheticWorkloadTest(
				const std::string& name,
				std::shared_ptr<Benchmarking::Workloads::SyntheticWorkload> workload,
				std::shared_ptr<Hardware::GPU> gpu,
				std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors)
				: Test(name, std::move(monitors))
				, workload_(std::move(workload))
				, gpu_(std::move(gpu)) {
			}
		}
	}
}