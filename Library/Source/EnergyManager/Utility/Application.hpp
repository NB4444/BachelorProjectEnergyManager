#pragma once

#include "EnergyManager/Hardware/CentralProcessor.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Utility/Logging/Loggable.hpp"
#include "EnergyManager/Utility/Runnable.hpp"

#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace EnergyManager {
	namespace Utility {
		/**
		 * An executable application.
		 */
		class Application : public Utility::Runnable {
			/**
			 * The path to the Application's main executable.
			 */
			std::string path_;

			/**
			 * The parameters to provide to the executable.
			 */
			std::vector<std::string> parameters_;

			/**
			 * The CPU affinity.
			 */
			std::vector<std::shared_ptr<Hardware::CentralProcessor>> affinity_;

			/**
			 * The GPU to use.
			 */
			std::shared_ptr<Hardware::GPU> gpu_;

			/**
			 * The output from the executable.
			 */
			std::string executableOutput_ = "";

			/**
			 * Whether to log output.
			 */
			bool logOutput_;

			/**
			 * Whether to inject the reporter application.
			 */
			bool injectReporter_;

			/**
			 * Whether or not to inject the EAR library.
			 */
			bool injectEAR_;

			/**
			 * The pipe of the current executable.
			 */
			pid_t processID_;

			/**
			 * The group ID to run with.
			 */
			int groupID_ = -1;

			/**
			 * The user ID to run with.
			 */
			int userID_ = -1;

		protected:
			std::vector<std::string> generateHeaders() const override;

			void beforeRun() final;

			void onRun() final;

		public:
			/**
			 * Creates a new Application.
			 * @param path The path to the Application's main executable.
			 * @param parameters The parameters to provide to the executable.
			 * @param affinity The CPU affinity.
			 * @param gpu The GPU to use.
			 * @param logOutput Whether to log the output.
			 * @param injectReporter Whether to inject the reporter application.
			 * @param injectEAR Whether to inject the EAR library.
			 */
			explicit Application(
				std::string path,
				std::vector<std::string> parameters = {},
				std::vector<std::shared_ptr<Hardware::CentralProcessor>> affinity = {},
				std::shared_ptr<Hardware::GPU> gpu = nullptr,
				const bool& logOutput = false,
				const bool& injectReporter = false,
				const bool& injectEAR = false);

			/**
			 * Copies the Application.
			 * @param application The Application to copy.
			 */
			Application(const Application& application);

			/**
			 * Gets the Application's path.
			 * @return The path.
			 */
			std::string getPath() const;

			/**
			 * Sets the Application's path.
			 * @param path The path.
			 */
			void setPath(const std::string& path);

			/**
			 * Gets the parameters to provide to the executable.
			 * @return The parameters.
			 */
			std::vector<std::string> getParameters() const;

			/**
			 * Sets the parameters to provide to the executable.
			 * @param parameters The parameters.
			 */
			void setParameters(const std::vector<std::string>& parameters);

			/**
			 * Gets the group ID to use to run the application.
			 * @return The group ID.
			 */
			int getGroupID();

			/**
			 * Sets the group ID to use to run the application.
			 * @param groupID The group ID.
			 */
			void setGroupID(const int& groupID);

			/**
			 * Gets the user ID to use to run the application.
			 * @return The user ID.
			 */
			int getUserID();

			/**
			 * Sets the user ID to use to run the application.
			 * @param userID The user ID.
			 */
			void setUserID(const int& userID);

			/**
			 * Retrieves the Application's affinity.
			 * @return The affinity.
			 */
			std::vector<std::shared_ptr<Hardware::CentralProcessor>> getAffinity() const;

			/**
			 * Sets the Application's affinity.
			 * @param affinity The affinity.
			 */
			void setAffinity(const std::vector<std::shared_ptr<Hardware::CentralProcessor>>& affinity);

			/**
			 * Gets the GPU to use.
			 * @return The GPU.
			 */
			std::shared_ptr<Hardware::GPU> getGPU() const;

			/**
			 * Sets the GPU to use.
			 * @param gpu The GPU.
			 */
			void setGPU(const std::shared_ptr<Hardware::GPU>& gpu);

			/**
			 * Gets whether to log the output.
			 * @return Whether to log the output.
			 */
			bool getLogOutput() const;

			/**
			 * Sets whether to log the output.
			 * @param logOutput Whether to log the output.
			 */
			void setLogOutput(const bool& logOutput);

			/**
			 * Gets the Application's output.
			 * @return The output.
			 */
			std::string getExecutableOutput() const;
		};
	}
}