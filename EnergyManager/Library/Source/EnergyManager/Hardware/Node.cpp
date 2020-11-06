#include "./Node.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"

#include <algorithm>
#include <numeric>
#include <sys/sysinfo.h>
#include <utility>

namespace EnergyManager {
	namespace Hardware {
		Node::Node(std::vector<std::shared_ptr<Hardware::CPU>> cpus, std::vector<std::shared_ptr<Hardware::GPU>> gpus) : cpus_(std::move(cpus)), gpus_(std::move(gpus)) {
		}

		std::shared_ptr<Node> Node::getNode() {
			static std::shared_ptr<Node> node = std::shared_ptr<Node>(new Node(CPU::getCPUs(), GPU::getGPUs()));

			return node;
		}

		Utility::Units::Byte Node::getMemorySize() const {
			struct sysinfo info;
			sysinfo(&info);

			return { info.totalram * info.mem_unit };
		}

		Utility::Units::Byte Node::getFreeMemorySize() const {
			struct sysinfo info;
			sysinfo(&info);

			return { info.freeram * info.mem_unit };
		}

		Utility::Units::Byte Node::getUsedMemorySize() const {
			return getMemorySize() - getFreeMemorySize();
		}

		Utility::Units::Byte Node::getSharedMemorySize() const {
			struct sysinfo info;
			sysinfo(&info);

			return { info.sharedram * info.mem_unit };
		}

		Utility::Units::Byte Node::getBufferMemorySize() const {
			struct sysinfo info;
			sysinfo(&info);

			return { info.bufferram * info.mem_unit };
		}

		Utility::Units::Byte Node::getSwapMemorySize() const {
			struct sysinfo info;
			sysinfo(&info);

			return { info.totalswap * info.mem_unit };
		}

		Utility::Units::Byte Node::getFreeSwapMemorySize() const {
			struct sysinfo info;
			sysinfo(&info);

			return { info.freeswap * info.mem_unit };
		}

		Utility::Units::Byte Node::getUsedSwapMemorySize() const {
			return getSwapMemorySize() - getFreeSwapMemorySize();
		}

		Utility::Units::Byte Node::getHighMemorySize() const {
			struct sysinfo info;
			sysinfo(&info);

			return { info.totalhigh * info.mem_unit };
		}

		Utility::Units::Byte Node::getFreeHighMemorySize() const {
			struct sysinfo info;
			sysinfo(&info);

			return { info.freehigh * info.mem_unit };
		}

		Utility::Units::Byte Node::getUsedHighMemorySize() const {
			return getHighMemorySize() - getFreeHighMemorySize();
		}

		unsigned int Node::getProcessCount() const {
			struct sysinfo info;
			sysinfo(&info);

			return info.procs;
		}

		Utility::Units::Joule Node::getEnergyConsumption() const {
			// Collect CPU energy consumptions
			std::vector<Utility::Units::Joule> cpuEnergyConsumptions;
			std::transform(cpus_.begin(), cpus_.end(), std::back_inserter(cpuEnergyConsumptions), [](auto& cpu) {
				return cpu->getEnergyConsumption();
			});

			// Collect GPU energy consumptions
			std::vector<Utility::Units::Joule> gpuEnergyConsumptions;
			std::transform(gpus_.begin(), gpus_.end(), std::back_inserter(gpuEnergyConsumptions), [](auto& gpu) {
				return gpu->getEnergyConsumption();
			});

			return std::accumulate(cpuEnergyConsumptions.begin(), cpuEnergyConsumptions.end(), Utility::Units::Joule())
				   + std::accumulate(gpuEnergyConsumptions.begin(), gpuEnergyConsumptions.end(), Utility::Units::Joule()) - startEnergyConsumption_;
		}

		Utility::Units::Watt Node::getPowerConsumption() const {
			// Collect CPU power consumptions
			std::vector<Utility::Units::Watt> cpuPowerConsumptions;
			std::transform(cpus_.begin(), cpus_.end(), std::back_inserter(cpuPowerConsumptions), [](auto& cpu) {
				return cpu->getPowerConsumption();
			});

			// Collect GPU power consumptions
			std::vector<Utility::Units::Watt> gpuPowerConsumptions;
			std::transform(gpus_.begin(), gpus_.end(), std::back_inserter(gpuPowerConsumptions), [](auto& gpu) {
				return gpu->getPowerConsumption();
			});

			return std::accumulate(cpuPowerConsumptions.begin(), cpuPowerConsumptions.end(), Utility::Units::Watt())
				   + std::accumulate(gpuPowerConsumptions.begin(), gpuPowerConsumptions.end(), Utility::Units::Watt());
		}
	}
}