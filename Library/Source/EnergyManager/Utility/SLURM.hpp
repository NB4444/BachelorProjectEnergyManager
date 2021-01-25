#pragma once

#include "EnergyManager/Configuration.hpp"
#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Environment.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"
#include "EnergyManager/Utility/Units/Hertz.hpp"

#include <string>

namespace EnergyManager {
	namespace Utility {
		namespace SLURM {
			/**
			 * The mutex used for job executions.
			 */
			static std::mutex mutex;

			/**
			 * Gets the ID of the current job.
			 * @return The ID.
			 */
			static unsigned int getCurrentJobID() {
				return Environment::getVariable<unsigned int>("SLURM_JOB_ID");
			}

			/**
			 * Gets the name of the current job.
			 * @return The name.
			 */
			static std::string getCurrentJobName() {
				return Environment::getVariable<std::string>("SLURM_JOB_NAME");
			}

			/**
			 * Gets the name of the current node.
			 * @return The name.
			 */
			static std::string getCurrentNodeName() {
				return Environment::getVariable<std::string>("SLURMD_NODENAME");
			}

			/**
			 * Gets all node names allocated to the current job.
			 * @return The node names.
			 */
			static std::vector<std::string> getNodeNames() {
				// Translate combined node list to individual hostnames
				Utility::Application scontrol(SLURM_SCONTROL, { "show", "hostnames", Environment::getVariable<std::string>("SLURM_JOB_NODELIST") });
				scontrol.run();

				// Get the node names and filter out empty ones
				auto nodeNames = Text::splitToVector(scontrol.getExecutableOutput(), "\n", true);
				std::vector<std::string> filteredNodeNames = {};
				for(const auto& nodeName : nodeNames) {
					if(nodeName != "") {
						filteredNodeNames.push_back(nodeName);
					}
				}

				return filteredNodeNames;
			}

			/**
			 * Gets the job ID that corresponds to the given job name.
			 * @param jobName The job name.
			 * @return The job ID.
			 */
			static unsigned int getJobID(const std::string& jobName) {
				// Get information about all SLURM jobs
				Utility::Application slurmJobInformationTool(SLURM_SCONTROL, { "show", "job" });
				slurmJobInformationTool.run(false);
				auto slurmJobInformation = slurmJobInformationTool.getExecutableOutput();

				// Parse the job ID
				int slurmJobID = -1;
				for(const auto& line : Utility::Text::splitToVector(slurmJobInformation, "\n")) {
					std::regex regex("JobId=(\\d+) JobName=" + jobName);
					std::smatch match;
					if(std::regex_search(line, match, regex)) {
						slurmJobID = std::stoul(match.str(1));

						Logging::logTrace("Found SLURM job ID %d", slurmJobID);
					}
				}
				if(slurmJobID < 0) {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("SLURM job not found");
				}

				return slurmJobID;
			}

			/**
			 * Logs information about the current job and node.
			 */
			static void logInformation() {
				Logging::logDebug(
					"SLURM job ID: %d\n"
					"SLURM job name: %s\n"
					//"SLURM hostname: %s\n"
					"SLURM node name: %s\n"
					"SLURM node list: %s",
					getCurrentJobID(),
					getCurrentJobName().c_str(),
					//Environment::getHostname().c_str(),
					getCurrentNodeName().c_str(),
					Text::join(getNodeNames(), ", ").c_str());
			}

			/**
			 * Creates a new SLURM job.
			 * @param applicationPath The path to the application to run through SLURM.
			 * @param applicationParameters The parameters to pass to the application.
			 * @param jobName The name of the SLURM job.
			 * @param nodes The amount of nodes to use.
			 * @param tasks The amount of tasks to run.
			 * @param exclusive Whether to reserve nodes exclusively for each task.
			 * @param minimumCPUFrequency The minimum CPU frequency to run at. Set to 0 to disable.
			 * @param maximumCPUFrequency The maximum CPU frequency to run at. Set to 0 to disable.
			 * @param gpuFrequency The GPU frequency to run at. Set to 0 to disable.
			 * @param ear Whether to integrate with EAR.
			 * @return The SLURM job.
			 */
			static auto runJob(
				const std::string& applicationPath,
				const std::vector<std::string>& applicationParameters,
				const std::string& jobName = "Job",
				const unsigned int& nodes = 1,
				const unsigned int& tasks = 1,
				const std::string& constraint = "",
				const bool& exclusive = true,
				const Units::Hertz& minimumCPUFrequency = Units::Hertz(),
				const Units::Hertz& maximumCPUFrequency = Units::Hertz(),
				const int& gpus = -1,
				const Units::Hertz& gpuFrequency = Units::Hertz(),
				const std::chrono::system_clock::duration& earMonitorInterval = std::chrono::milliseconds(50),
				const bool& verbose = false) {
				std::lock_guard<std::mutex> lock(mutex);

				Logging::logDebug("Creating SLURM job...");

				auto slurmGPUFrequency = static_cast<unsigned int>(gpuFrequency.convertPrefix(Units::SIPrefix::MEGA));

				// Write the job script
				std::string command
					= "#!/bin/bash\n"
					  "#SBATCH --output %J.log\n"
					  "#SBATCH --error %J.log\n"
					+ std::string(verbose ? "#SBATCH --verbose\n" : "")
					+ "#SBATCH --job-name EnergyManager-JobRunner\n"
					  "#SBATCH --account COLBSC\n"
					  "#SBATCH --partition standard\n"
					  "#SBATCH --time 1:00:00\n"
					  "#SBATCH --constraint "
					+ constraint
					+ "\n"
					  "#SBATCH --nodes "
					+ Text::toString(nodes)
					+ "\n"
					  "#SBATCH --ntasks "
					+ Text::toString(tasks)
					+ "\n"
					  "#SBATCH --ntasks-per-node 1\n"
					  "#SBATCH --exclusive\n"

					  "module purge\n"
					  "source /hpc/base/ctt/bin/setup_modules.sh\n"
					  "module load base-env\n"
					  "module load git/2.6.2\n"
					  "module load compiler/intel/20.4\n"
					  "module load compiler/gnu/8.2.0\n"
					  "module load cmake/3.14.5\n"
					  "module load cuda/10.1\n"
					  "module load mpi/openmpi/4.0.5_gnu\n"
					  "module load boost/1.70.0/impi\n"
					  "module load ear/ear\n"

#ifdef EAR_ENABLED
					  // Load the EAR library
					  //  "export SLURM_HACK_EARL_INSTALL_PATH="
					  //+ std::string(EAR_LIBRARY_DIRECTORIES)
					  //+ "\n"
					  //  "export SLURM_HACK_LOADER="
					  //+ std::string(EAR_DAEMON_LIBRARIES)
					  //+ "\n"
					  //  "export SLURM_HACK_LIBRARY_FILE="
					  //+ std::string(EAR_LIBRARIES)
					  //+ "\n"
					  "export SLURM_HACK_EARL_VERBOSE=2\n"
					  "export SLURM_LOADER_LOAD_NO_MPI_LIB=\"$@\"\n"
#endif

					//// Set the GPU frequency in EAR if necessary
					//// EAR requires the value to be in MHz
					//+ (gpuFrequency == Units::Hertz() ? "" : "export SLURM_EAR_GPU_DEF_FREQ=" + Text::toString(static_cast<unsigned int>(gpuFrequency.convertPrefix(Units::SIPrefix::MEGA))) + "\n")

					+ "srun"

					  // Set the CPU frequency in SLURM
					  // SLURM want the CPU frequency in KHz
					  " --cpu-freq "
					+ (maximumCPUFrequency == Units::Hertz()
						   ? "2600000"
						   : ((minimumCPUFrequency == Units::Hertz() ? "" : (Text::toString(static_cast<unsigned int>(minimumCPUFrequency.convertPrefix(Units::SIPrefix::KILO))) + "-"))
							  + Text::toString(static_cast<unsigned int>(maximumCPUFrequency.convertPrefix(Units::SIPrefix::KILO)))))

					+ (gpus >= 0 ? " --gpus " + Text::toString(gpus) : "")

					// Set the GPU frequency in SLURM
					// SLURM wants the GPU frequency in MHz
					+ (gpuFrequency == Units::Hertz() ? "" : (" --gpu-freq " + (slurmGPUFrequency == 0 ? "low" : Text::toString(slurmGPUFrequency))))

					+ (verbose ? " --verbose" : "")
					+ " --job-name EnergyManager"
					  //#ifdef EAR_ENABLED
					  //					  " --ear-verbose 1"
					  //					  " --ear-policy monitoring"
					  //#endif
					  " \""
					+ applicationPath + "\"" + (applicationParameters.empty() ? "" : (" \"" + Text::join(applicationParameters, "\" \"") + "\""));
				std::ofstream jobScriptOutput("JobRunner.sh.tmp");
				jobScriptOutput << command;
				jobScriptOutput.close();

				// Run the job script
				system("chmod +x JobRunner.sh.tmp");
				Application sbatch(SLURM_SBATCH, { "--wait", "JobRunner.sh.tmp" });
				sbatch.setLogOutput(true);
				sbatch.run();
				auto sbatchOutput = sbatch.getExecutableOutput();

				// Capture SLURM job IDs if available
				std::regex regex("Submitted batch job (\\d+)");
				std::smatch match;
				unsigned int jobID;
				if(std::regex_search(sbatchOutput, match, regex)) {
					jobID = std::stoul(match.str(1));

					Logging::logDebug("Found SLURM job ID %d", jobID);
				} else {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Job ID not found in executable output:\n" + sbatchOutput);
				}

				// Capture the output
				std::string jobOutput;
				Exceptions::Exception::retry(
					[&] {
						jobOutput = Text::readFile(Text::toString(jobID) + ".log");

						if(jobOutput.empty()) {
							ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not retrieve job output");
						}
					},
					10,
					std::chrono::seconds(1));
				Logging::logTrace("SLURM job output: %s", jobOutput.c_str());

				// Return the results
				struct {
					unsigned int jobID;
					std::string output;
				} result { .jobID = jobID, .output = jobOutput };
				return result;
			}
		}
	}
}