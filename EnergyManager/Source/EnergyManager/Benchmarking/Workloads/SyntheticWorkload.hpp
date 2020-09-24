#pragma once

#include "EnergyManager/Benchmarking/Operations/SyntheticOperation.hpp"
#include "EnergyManager/Hardware/GPU.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class SyntheticWorkload {
				using Parser = std::function<std::shared_ptr<SyntheticWorkload>(const std::string& name, const std::map<std::string, std::string>& parameters)>;

				static std::vector<Parser> parsers_;

				std::vector<std::shared_ptr<Operations::SyntheticOperation>> operations_;

			protected:
				static void addParser(const Parser& parser);

			public:
				static std::shared_ptr<SyntheticWorkload> parse(const std::string& name, const std::map<std::string, std::string>& parameters);

				SyntheticWorkload(std::vector<std::shared_ptr<Operations::SyntheticOperation>> operations = {});

				void addOperation(const std::shared_ptr<Operations::SyntheticOperation>& operation);

				void run(const std::shared_ptr<Hardware::GPU>& gpu);
			};
		}
	}
}