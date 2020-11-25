#pragma once

#include "EnergyManager/Hardware/CentralProcessor.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Persistence/Entity.hpp"
#include "EnergyManager/Utility/Logging/Loggable.hpp"
#include "EnergyManager/Utility/Runnable.hpp"

#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace EnergyManager {
	namespace Testing {
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
			 * The pipe of the current executable.
			 */
			pid_t processID_;

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
			 */
			explicit Application(
				std::string path,
				std::vector<std::string> parameters = {},
				std::vector<std::shared_ptr<Hardware::CentralProcessor>> affinity = {},
				std::shared_ptr<Hardware::GPU> gpu = nullptr);

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
			 * Gets the Application's output.
			 * @return The output.
			 */
			std::string getExecutableOutput() const;
		};
	}
}