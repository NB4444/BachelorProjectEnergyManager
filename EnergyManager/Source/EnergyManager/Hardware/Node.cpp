#include "./Node.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"

#include <sys/sysinfo.h>

namespace EnergyManager {
	namespace Hardware {
		Node::Node(const std::shared_ptr<Hardware::CPU>& cpu, const std::shared_ptr<Hardware::GPU>& gpu) : cpu_(cpu), gpu_(gpu) {
			startEnergyConsumption_ = getEnergyConsumption();
		}

		std::shared_ptr<Node> Node::getNode() {
			// TODO: Detect all CPUs and GPUs
			static std::shared_ptr<Node> node = std::shared_ptr<Node>(new Node(CPU::getCPU(0), GPU::getGPU(0)));

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
			return cpu_->getEnergyConsumption() + gpu_->getEnergyConsumption() - startEnergyConsumption_;
		}

		Utility::Units::Watt Node::getPowerConsumption() const {
			return cpu_->getPowerConsumption() + gpu_->getPowerConsumption();
		}
	}
}