#include "./EARMonitor.hpp"

#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/SLURM.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> EARMonitor::getEARValues() const {
				if(slurmJobID_ < 0) {
					return {};
				}

				const auto csvFileName = "ear.csv.tmp";

				// Contact the EAR accounting tool and store the data in CSV format
				logTrace("Looking for EAR job with ID %d.%d", slurmJobID_, slurmStepID_);
				Utility::Application earAccountingTool(EAR_EACCT, { "-j", Utility::Text::toString(slurmJobID_) + "." + Utility::Text::toString(slurmStepID_), "-c", csvFileName });
				earAccountingTool.run(false);
				auto output = Utility::Text::trim(earAccountingTool.getExecutableOutput());

				if(output == "No jobs found.") {
					logTrace("No data available yet");
					return {};
				} else {
					logTrace("Found EAR data, loading...");
					const auto earData = Utility::Text::readFile(csvFileName);

					logTrace("Parsing EAR data...");
					const auto data = Utility::Text::parseTable(earData, "\n", ";");

					logTrace("EAR data: {%s}", Utility::Text::join(data[0], ", ", ": ").c_str());

					return data[0];
				}
			}

			void EARMonitor::beforeRun() {
				// Wait for the job and step to show up in EAR
				while(getEARValues().empty()) {
					sleep(std::chrono::milliseconds(10));
				}

				logDebug("EAR data available, starting polling loop...");
			}

			std::map<std::string, std::string> EARMonitor::onPoll() {
				const auto values = getEARValues();

				std::map<std::string, std::string> results
					= { { "applicationRuntime",
						  Utility::Text::toString(
							  std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(static_cast<long>(std::stod(values.at("TIME(s)")) * 1000.0))).count()) },
						{ "cpuCoreClockRate", Utility::Text::toString(Utility::Units::Hertz(std::stod(values.at("FREQ(GHz)")), Utility::Units::SIPrefix::GIGA).toValue()) },
						{ "averagePowerConsumption", Utility::Text::toString(Utility::Units::Watt(std::stod(values.at("POWER(Watts)"))).toValue()) },
						{ "energyConsumption", Utility::Text::toString(Utility::Units::Joule(std::stod(values.at("ENERGY(J)"))).toValue()) } };

				return results;
			}

			EARMonitor::EARMonitor(const unsigned int& slurmJobID, const unsigned int& slurmStepID, const std::chrono::system_clock::duration& interval)
				: Monitor("EARMonitor", interval)
				, slurmJobID_(slurmJobID)
				, slurmStepID_(slurmStepID) {
			}

			EARMonitor::EARMonitor(const unsigned int& slurmStepID, const std::chrono::system_clock::duration& interval) : EARMonitor(Utility::SLURM::getCurrentJobID(), slurmStepID, interval) {
			}

			EARMonitor::EARMonitor(std::string slurmJobName, const unsigned int& slurmStepID, const std::chrono::system_clock::duration& interval)
				: EARMonitor(Utility::SLURM::getJobID(slurmJobName), slurmStepID, interval) {
			}

			unsigned int EARMonitor::getSLURMJobID() const {
				return slurmJobID_;
			}

			void EARMonitor::setSLURMJobID(const unsigned int& slurmJobID) {
				slurmJobID_ = slurmJobID;
			}
		}
	}
}