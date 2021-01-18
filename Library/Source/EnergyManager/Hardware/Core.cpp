#include "./Core.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Utility/CachedValue.hpp"
#include "EnergyManager/Utility/Collections.hpp"
#include "EnergyManager/Utility/EAR.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <chrono>

namespace EnergyManager {
	namespace Hardware {
		std::chrono::system_clock::duration Core::getProcStatTimespan(const std::string& name) const {
			static Utility::CachedValue<std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>>> procStatTimespansPerCPU(
				std::chrono::milliseconds(100));

			return procStatTimespansPerCPU
				.getValue([&](const auto& value, const auto& timeSinceLastUpdate) {
					std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> result;

					const auto& currentProcStatValues = getProcStatValuesPerCPU();

					for(const auto& cpuProcStatData : currentProcStatValues) {
						const auto& cpuID = cpuProcStatData.first;
						const auto& cpu = CPU::getCPU(cpuID);

						for(const auto& coreProcStatData : cpuProcStatData.second) {
							const auto& coreID = coreProcStatData.first;
							const auto& data = coreProcStatData.second;

							for(const auto& timespanData : data) {
								const auto& timespanName = timespanData.first;
								const auto& timespan = timespanData.second;

								const auto difference = timespan - cpu->startProcStatValues_.at(cpuID).at(coreID).at(timespanName);

								result[cpuID][coreID][timespanName] = difference >= std::chrono::system_clock::duration(0) ? difference : std::chrono::system_clock::duration(0);
							}
						}
					}

					return result;
				})
				.at(cpu_->getID())
				.at(getID())
				.at(name);
		}

		std::shared_ptr<Core> Core::getCore(CPU* cpu, const unsigned int& id) {
			// Keep track of Cores
			static std::map<uint32_t, std::map<uint32_t, std::shared_ptr<Core>>> allCPUCores;

			// Only allow one thread to get Cores at a time
			static std::mutex mutex;
			std::lock_guard<std::mutex> guard(mutex);

			// Allocate space if necessary
			if(allCPUCores.find(cpu->getID()) == allCPUCores.end()) {
				allCPUCores[cpu->getID()] = {};
			}

			// Get the current CPU cores list
			auto& cpuCores = allCPUCores.at(cpu->getID());

			// Create the core if it does not exist yet
			if(cpuCores.find(id) == cpuCores.end()) {
				cpuCores[id] = std::shared_ptr<Core>(new Core(cpu, Utility::Collections::getMapItemByIndex(getProcCPUInfoValuesPerCPU().at(cpu->getID()), id).first, id));
			}

			return cpuCores.at(id);
		}

		Core::Core(CPU* cpu, const unsigned int& id, const unsigned int& coreID)
			: CentralProcessor(id)
			, cpu_(cpu)
			, coreID_(coreID)
			, lastUtilizationRateUpdateProcStatValues_(getProcStatValuesPerCPU().at(cpu->getID())) {
		}

		std::vector<std::string> Core::generateHeaders() const {
			return { "CPU " + Utility::Text::toString(cpu_->getID()), "Core " + Utility::Text::toString(getID()) };
		}

		std::shared_ptr<Core> Core::getCore(const unsigned int& id) {
			for(const auto& cpu : CPU::getCPUs()) {
				for(const auto& core : cpu->getCores()) {
					if(core->getID() == id) {
						return core;
					}
				}
			}

			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not find core");
		}

		std::shared_ptr<CPU> Core::getCPU() const {
			return CPU::getCPU(cpu_->getID());
		}

		unsigned int Core::getCoreID() const {
			return coreID_;
		}

		Utility::Units::Hertz Core::getCoreClockRate() const {
			std::ifstream coreClockRateStream("/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/scaling_cur_freq");
			std::string coreClockRateString((std::istreambuf_iterator<char>(coreClockRateStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Hertz(std::stoul(coreClockRateString), Utility::Units::SIPrefix::KILO);
		}

		Utility::Units::Hertz Core::getCurrentMinimumCoreClockRate() const {
			if(getPowerScalingDriver() == "intel_pstate") {
				std::ifstream minimumCoreClockRateStream("/sys/devices/system/cpu/intel_pstate/min_perf_pct");
				std::string minimumCoreClockRateString((std::istreambuf_iterator<char>(minimumCoreClockRateStream)), std::istreambuf_iterator<char>());
				const auto minimumCoreClockRatePercentage = std::stod(minimumCoreClockRateString);

				return Utility::Units::Hertz(minimumCoreClockRatePercentage / 100 * getMaximumCoreClockRate().toValue());
			} else {
				std::ifstream minimumCoreClockRateStream("/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/scaling_min_freq");
				std::string minimumCoreClockRateString((std::istreambuf_iterator<char>(minimumCoreClockRateStream)), std::istreambuf_iterator<char>());

				return Utility::Units::Hertz(std::stoul(minimumCoreClockRateString), Utility::Units::SIPrefix::KILO);
			}
		}

		Utility::Units::Hertz Core::getCurrentMaximumCoreClockRate() const {
			if(getPowerScalingDriver() == "intel_pstate") {
				return Utility::Units::Hertz(std::stod(Utility::Text::readFile("/sys/devices/system/cpu/intel_pstate/max_perf_pct")) / 100 * getMaximumCoreClockRate().toValue());
			} else {
				return Utility::Units::Hertz(
					std::stoul(Utility::Text::readFile("/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/scaling_max_freq")),
					Utility::Units::SIPrefix::KILO);
			}
		}

		void Core::setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) {
			logDebug("Setting clock rate range to [%lu, %lu]...", minimumRate.toValue(), minimumRate.toValue());

#ifdef EAR_ENABLED
			Utility::EAR::setCoreClockRates({ shared_from_this() }, maximumRate);
#else
			// Set minimum rate
			Utility::Text::writeFile(
				"/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/scaling_min_freq",
				Utility::Text::toString(minimumRate.convertPrefix(Utility::Units::SIPrefix::KILO)));

			// Set maximum rate
			Utility::Text::writeFile(
				"/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/scaling_max_freq",
				Utility::Text::toString(maximumRate.convertPrefix(Utility::Units::SIPrefix::KILO)));
#endif
		}

		void Core::resetCoreClockRate() {
			logDebug("Resetting clock rate...");

			setCoreClockRate(
				Utility::Units::Hertz(
					std::stoul(Utility::Text::readFile("/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/cpuinfo_min_freq")),
					Utility::Units::SIPrefix::KILO),
				Utility::Units::Hertz(
					std::stoul(Utility::Text::readFile("/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/cpuinfo_max_freq")),
					Utility::Units::SIPrefix::KILO));
		}

		Utility::Units::Percent Core::getCoreUtilizationRate() {
			// Cache the value so we don't keep calculating this again
			static Utility::CachedValue<std::map<unsigned int, std::map<unsigned int, Utility::Units::Percent>>> cpuCoreUtilizationRates(std::chrono::milliseconds(100));

			return cpuCoreUtilizationRates
				.getValue([&](const auto& value, const auto& timeSinceLastUpdate) {
					std::map<unsigned int, std::map<unsigned int, Utility::Units::Percent>> result;

					// Get the current values
					const auto currentProcStatValues = getProcStatValuesPerCPU();
					static auto lastProcStatValues = currentProcStatValues;

					for(const auto& cpuProcStatData : currentProcStatValues) {
						const auto& cpu = cpuProcStatData.first;

						for(const auto& coreProcStatData : cpuProcStatData.second) {
							const auto& core = coreProcStatData.first;

							// Calculate the core utilization rates
							const auto lastCoreValues = lastProcStatValues.at(cpu).at(core);
							const auto previousIdle = lastCoreValues.at("idleTimespan") + lastCoreValues.at("ioWaitTimespan");
							const auto previousActive = lastCoreValues.at("userTimespan") + lastCoreValues.at("niceTimespan") + lastCoreValues.at("systemTimespan")
														+ lastCoreValues.at("interruptsTimespan") + lastCoreValues.at("softInterruptsTimespan") + lastCoreValues.at("stealTimespan")
														+ lastCoreValues.at("guestTimespan") + lastCoreValues.at("guestNiceTimespan");
							const auto previousTotal = previousIdle + previousActive;

							const auto currentCoreValues = coreProcStatData.second;
							const auto idle = currentCoreValues.at("idleTimespan") + currentCoreValues.at("ioWaitTimespan");
							const auto active = currentCoreValues.at("userTimespan") + currentCoreValues.at("niceTimespan") + currentCoreValues.at("systemTimespan")
												+ currentCoreValues.at("interruptsTimespan") + currentCoreValues.at("softInterruptsTimespan") + currentCoreValues.at("stealTimespan")
												+ currentCoreValues.at("guestTimespan") + currentCoreValues.at("guestNiceTimespan");
							const auto total = idle + active;

							const auto totalDifference = total - previousTotal;
							const auto idleDifference = idle - previousIdle;
							const auto activeDifference = active - previousActive;

							// Add the value
							result[cpu][core]
								= Utility::Units::Percent(totalDifference.count() == 0 ? 0 : (static_cast<double>(activeDifference.count()) / static_cast<double>(totalDifference.count()) * 100.0));
						}
					}

					lastProcStatValues = currentProcStatValues;

					return result;
				})
				.at(getCPU()->getID())
				.at(getID());
		}

		Utility::Units::Hertz Core::getMaximumCoreClockRate() const {
			std::ifstream inputStream("/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/cpuinfo_max_freq");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Hertz(std::stoi(cpuInfo), Utility::Units::SIPrefix::KILO);
		}

		Utility::Units::Hertz Core::getMinimumCoreClockRate() const {
			std::ifstream inputStream("/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/cpuinfo_min_freq");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Hertz(std::stoi(cpuInfo), Utility::Units::SIPrefix::KILO);
		}

		Utility::Units::Joule Core::getEnergyConsumption() {
			// TODO: Maybe there is a way to measure the energy consumption of individual cores
			return getCPU()->getEnergyConsumption() / getCPU()->getCores().size();
		}

		Utility::Units::Watt Core::getPowerConsumption() {
			// First calculate the total utilization
			double totalUtilization = 0;
			for(const auto& core : getCPU()->getCores()) {
				totalUtilization += core->getCoreUtilizationRate().getUnit();
			}

			// Now we can determine the power consumption per core by using the assumption that it is proportional to the utilization rate
			return totalUtilization == 0 ? 0 : ((getCPU()->getPowerConsumption().toValue() / totalUtilization) * getCoreUtilizationRate().getUnit());
		}

		Utility::Units::Celsius Core::getTemperature() const {
			// TODO: Maybe there is a way to measure the temperature of individual cores
			return getCPU()->getTemperature();
		}

		std::chrono::system_clock::duration Core::getUserTimespan() const {
			return getProcStatTimespan("userTimespan");
		}

		std::chrono::system_clock::duration Core::getNiceTimespan() const {
			return getProcStatTimespan("niceTimespan");
		}

		std::chrono::system_clock::duration Core::getSystemTimespan() const {
			return getProcStatTimespan("systemTimespan");
		}

		std::chrono::system_clock::duration Core::getIdleTimespan() const {
			return getProcStatTimespan("idleTimespan");
		}

		std::chrono::system_clock::duration Core::getIOWaitTimespan() const {
			return getProcStatTimespan("ioWaitTimespan");
		}

		std::chrono::system_clock::duration Core::getInterruptsTimespan() const {
			return getProcStatTimespan("interruptsTimespan");
		}

		std::chrono::system_clock::duration Core::getSoftInterruptsTimespan() const {
			return getProcStatTimespan("softInterruptsTimespan");
		}

		std::chrono::system_clock::duration Core::getStealTimespan() const {
			return getProcStatTimespan("stealTimespan");
		}

		std::chrono::system_clock::duration Core::getGuestTimespan() const {
			return getProcStatTimespan("guestTimespan");
		}

		std::chrono::system_clock::duration Core::getGuestNiceTimespan() const {
			return getProcStatTimespan("guestNiceTimespan");
		}

		std::string Core::getPowerScalingDriver() const {
			return Utility::Text::trim(Utility::Text::readFile("/sys/devices/system/cpu/cpu" + Utility::Text::toString(getID()) + "/cpufreq/scaling_driver"));
		}
	}
}