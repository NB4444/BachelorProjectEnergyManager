#pragma once

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
				const Units::Hertz& gpuFrequency = Units::Hertz(),
				const bool& ear = false,
				const std::chrono::system_clock::duration& earMonitorInterval = std::chrono::milliseconds(50)) {
				Logging::logDebug("Creating SLURM job...");

				// Write the job script
				std::string command
					= "#!/bin/bash\n"
					  "#SBATCH --output log.%J.out\n"
					  "#SBATCH --error log.%J.out\n"
					  "#SBATCH --verbose\n"
					  "#SBATCH --job-name EnergyManager-JobRunner\n"
					  "#SBATCH --account COLBSC\n"
					  "#SBATCH --partition standard\n"
					  "#SBATCH --time 1:00:00\n"
					  "#SBATCH --constraint " + constraint + "\n"
					  "#SBATCH --nodes " + Text::toString(nodes) + "\n"
					  "#SBATCH --ntasks " + Text::toString(tasks) + "\n"
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
					  "module load boost/1.67.0/impi\n"
					  "module load ear/ear\n"

					  // Load the EAR library
					  "export LD_PRELOAD=${EAR_INSTALL_PATH}/lib/libear.seq.so\n"
					  "export SLURM_HACK_LIBRARY_FILE=${EAR_INSTALL_PATH}/lib/libear.seq.so\n"

					  // Set the GPU frequency if necessary
					  + (gpuFrequency == Units::Hertz()
						 ? ""
						 : "export SLURM_EAR_GPU_DEF_FREQ=\"$3\"\n")

					  + "srun"
					  + (maximumCPUFrequency == Units::Hertz()
						 ? ""
						 : (" --cpu-freq " + (minimumCPUFrequency == Units::Hertz()
								 ? ""
								 : (Text::toString(minimumCPUFrequency.convertPrefix(Units::SIPrefix::MEGA)) + "-")) + Text::toString(maximumCPUFrequency.convertPrefix(Units::SIPrefix::MEGA))))
					  + (gpuFrequency == Units::Hertz()
						 ? ""
						 : (" --gpu-freq " + Text::toString(gpuFrequency.convertPrefix(Units::SIPrefix::MEGA))))
					  + " --verbose"
					  " --job-name EnergyManager"
					  " --ear-verbose 1"
					  " --ear-policy monitoring"
					  " \"" + applicationPath + "\""
					  + (applicationParameters.empty() ? "" : (" \"" + Text::join(applicationParameters, "\" \"") + "\""));
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
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Job ID not found");
				}

				// Capture the output
				auto jobOutput = Text::readFile("log." + Text::toString(jobID) + ".out");
				Logging::logTrace("SLURM job output: %s", jobOutput.c_str());

				// Return the results
				struct {
					unsigned int jobID;
					std::string output;
				} result { .jobID = jobID, .output = jobOutput };
				return result;
				//// Prepare the SLURM parameters
				//std::vector<std::string> srunParameters;
				//
				//// Enable verbose mode
				//srunParameters.push_back("--verbose");
				//
				//// Set the name
				//srunParameters.push_back("--job-name");
				//srunParameters.push_back(jobName);
				//
				////// Set the time limit
				////slurmParameters.push_back("--time");
				////slurmParameters.push_back("1:00:00");
				//
				//// Ensure exclusive access
				//if(exclusive) {
				//	srunParameters.push_back("--exclusive");
				//}
				//
				////// Set the partition
				////slurmParameters.push_back("--partition");
				////slurmParameters.push_back("standard");
				//
				////// Set the account
				////slurmParameters.push_back("--account");
				////slurmParameters.push_back("COLBSC");
				//
				//// Configure amount of nodes
				//srunParameters.push_back("--nodes");
				//srunParameters.push_back(Text::toString(nodes));
				//
				//// Set the amount of tasks
				//srunParameters.push_back("--ntasks");
				//srunParameters.push_back(Text::toString(tasks));
				//
				//if(constraint != "") {
				//	srunParameters.push_back("--constraint");
				//	srunParameters.push_back(constraint);
				//}
				//
				////// Set the amount of repetitions per node
				////slurmParameters.push_back("--ntasks-per-node");
				////slurmParameters.push_back("1");
				//
				////// Constrain the nodes from which to select by using features
				////slurmParameters.push_back("--constraint");
				////slurmParameters.push_back("V100_16GB&NUMGPU2&rack26&EDR");
				////slurmParameters.push_back("2666MHz&NUMGPU2&V100_16GB");
				//
				////// Set up the output file
				////slurmParameters.push_back("-o");
				////slurmParameters.push_back("log.%j.profiler-session.out");
				////
				////// Set up the error output file
				////slurmParameters.push_back("-e");
				////slurmParameters.push_back("log.%j.profiler-session.err");
				//
				//// Set the CPU frequency
				//if(maximumCPUFrequency != Units::Hertz()) {
				//	srunParameters.push_back("--cpu-freq");
				//	srunParameters.push_back(
				//		(minimumCPUFrequency == Units::Hertz() ? "" : (Text::toString(minimumCPUFrequency.convertPrefix(Units::SIPrefix::MEGA)) + "-"))
				//		+ Text::toString(maximumCPUFrequency.convertPrefix(Units::SIPrefix::MEGA)));
				//}
				//
				//// Set the GPU frequency
				//if(gpuFrequency != Units::Hertz()) {
				//	srunParameters.push_back("--gpu-freq");
				//	srunParameters.push_back(Text::toString(gpuFrequency.convertPrefix(Units::SIPrefix::MEGA)));
				//}
				//
				//// Configure the EAR framework
				//if(ear) {
				//	// Inject EAR into SLURM
				//	const auto earInstallPath = Environment::getVariable<std::string>("EAR_INSTALL_PATH");
				//	Environment::setVariable("LD_PRELOAD", earInstallPath + "/lib/libear.seq.so");
				//	Environment::setVariable("SLURM_HACK_LIBRARY_FILE", earInstallPath + "/lib/libear.seq.so");
				//
				//	// Disable loading MPI
				//	Environment::setVariable("SLURM_LOADER_LOAD_NO_MPI_LIB", applicationPath);
				//
				//	// Set the verbosity
				//	srunParameters.push_back("--ear-verbose");
				//	srunParameters.push_back("1");
				//
				//	// Set the policy
				//	srunParameters.push_back("--ear-policy");
				//	srunParameters.push_back("monitoring");
				//
				//	// Set the GPU frequency
				//	if(gpuFrequency != Units::Hertz()) {
				//		Environment::setVariable("SLURM_EAR_GPU_DEF_FREQ", Text::toString(gpuFrequency.convertPrefix(Units::SIPrefix::MEGA)));
				//	}
				//}
				//
				//// Combine the parameters
				//srunParameters.push_back(applicationPath);
				//srunParameters.insert(srunParameters.end(), applicationParameters.begin(), applicationParameters.end());
				//
				//// Prepare and return the SLURM runner
				//auto job = Application(SLURM_SALLOC, srunParameters);
				//job.setLogOutput(true);
				//job.run();
				//
				//return job.getExecutableOutput();
			}
		}
	}
}