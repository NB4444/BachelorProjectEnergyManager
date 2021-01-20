#include "./Node.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Utility/ProtectedMakeShared.hpp"

#include <algorithm>
#include <numeric>
#include <sys/sysinfo.h>
#include <utility>

namespace EnergyManager {
	namespace Hardware {
		Node::Node(std::vector<std::shared_ptr<Hardware::CPU>> cpus, std::vector<std::shared_ptr<Hardware::GPU>> gpus) : cpus_(std::move(cpus)), gpus_(std::move(gpus)) {
		}

		std::shared_ptr<Node> Node::getNode() {
			// Only allow one thread to get Nodes at a time
			static std::mutex mutex;
			std::lock_guard<std::mutex> guard(mutex);

			static std::shared_ptr<Node> node = Utility::protectedMakeShared<Node>(CPU::getCPUs(), GPU::getGPUs());

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

		Utility::Units::Joule Node::getEnergyConsumption() {
			double energyConsumption = 0.0;

			// Collect CPU energy consumptions
			for(auto& cpu : cpus_) {
				energyConsumption += cpu->getEnergyConsumption().toValue();
			}

			// Collect GPU energy consumptions
			for(auto& gpu : gpus_) {
				energyConsumption += gpu->getEnergyConsumption().toValue();
			}

			return Utility::Units::Joule(energyConsumption) - startEnergyConsumption_;
		}

		Utility::Units::Watt Node::getPowerConsumption() {
			double powerConsumption = 0.0;

			// Collect CPU power consumptions
			for(auto& cpu : cpus_) {
				powerConsumption += cpu->getPowerConsumption().toValue();
			}

			// Collect GPU power consumptions
			for(auto& gpu : gpus_) {
				powerConsumption += gpu->getPowerConsumption().toValue();
			}

			return Utility::Units::Watt(powerConsumption);
		}
	}
}