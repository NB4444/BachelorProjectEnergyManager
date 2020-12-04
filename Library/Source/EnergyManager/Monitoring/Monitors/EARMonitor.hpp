#pragma once

#include "EnergyManager/Hardware/Node.hpp"
#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"
#include "EnergyManager/Utility/Units/Joule.hpp"

#include <mutex>

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			/**
			 * Monitors a Node.
			 */
			class EARMonitor : public Monitor {
				/**
				 * The mutex used for the monitor.
				 */
				static std::mutex mutex_;

				/**
				 * The ID of the SLURM job to monitor.
				 */
				unsigned int slurmJobID_;

				/**
				 * The ID of the step within the SLURM job that ran EAR to monitor.
				 */
				unsigned int slurmStepID_;

				/**
				 * Collects the values from EAR.
				 * @return The values.
				 */
				std::map<std::string, std::string> getEARValues() const;

			protected:
				void beforeRun() override;

				std::map<std::string, std::string> onPoll() final;

			public:
				/**
				 * Creates a new EARMonitor.
				 * @param slurmJobID The ID of the SLURM job that ran EAR to monitor.
				 * @param slurmStepID The ID of the step within the SLURM job that ran EAR to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				EARMonitor(const unsigned int& slurmJobID, const unsigned int& slurmStepID, const std::chrono::system_clock::duration& interval);

				/**
				 * Creates a new EARMonitor for the current SLURM job.
				 * @param slurmStepID The ID of the step within the SLURM job that ran EAR to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				EARMonitor(const unsigned int& slurmStepID, const std::chrono::system_clock::duration& interval);

				/**
				 * Creates a new EARMonitor.
				 * @param slurmJobName The name of the SLURM job that ran EAR to monitor.
				 * @param slurmStepID The ID of the step within the SLURM job that ran EAR to monitor.
				 * @param interval The interval at which to poll the monitored variables.
				 */
				EARMonitor(std::string slurmJobName, const unsigned int& slurmStepID, const std::chrono::system_clock::duration& interval);

				/**
				 * Gets the job ID to look for.
				 * @return The job ID.
				 */
				unsigned int getSLURMJobID() const;

				/**
				 * Sets the job ID to look for.
				 * @param jobID The job ID.
				 */
				void setSLURMJobID(const unsigned int& slurmJobID);
			};
		}
	}
}