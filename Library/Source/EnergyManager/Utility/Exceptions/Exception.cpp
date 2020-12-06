#include "./Exception.hpp"

#include "EnergyManager/Utility/Logging.hpp"

#include <boost/filesystem.hpp>
#include <csignal>
#include <unistd.h>
#include <utility>

namespace EnergyManager {
	namespace Utility {
		namespace Exceptions {
			const std::string Exception::backtraceFile_ = "backtrace.dump";

			StaticInitializer Exception::signalHandlerInitializer_ = StaticInitializer([] {
				// Register the signal handler
				//::signal(SIGSEGV, &signalHandler);
				//::signal(SIGABRT, &signalHandler);

				// Check if there was a crash on the last startup
				if(boost::filesystem::exists(backtraceFile_)) {
					// There was a crash and backtrace
					{
						// Open the crash dump
						std::ifstream input(backtraceFile_);

						// Extract the stacktrace and display it
						auto stacktrace = boost::stacktrace::stacktrace::from_dump(input);
						Logging::logWarning("Previous run appears to have crashed with the following stacktrace:\n%s", boost::stacktrace::to_string(stacktrace).c_str());
					}

					// Clean up the file
					boost::filesystem::remove(backtraceFile_);
				}
			});

			void Exception::signalHandler(int signalNumber) {
				::signal(signalNumber, SIG_DFL);
				boost::stacktrace::safe_dump_to(backtraceFile_.c_str());
				Logging::logError("Program halted with signal number %d\n\nStacktrace:\n%s", __FILE__, __LINE__, signalNumber, boost::stacktrace::to_string(boost::stacktrace::stacktrace()).c_str());
				exit(signalNumber);
			}

			void Exception::retry(const std::function<void()>& operation, const unsigned int& attempts, const std::chrono::system_clock::duration& attemptInterval) {
				for(unsigned int attempt = 1; attempts == 0 || attempt <= attempts; ++attempt) {
					// Wait before retrying
					if(attempt > 1) {
						Runnable::sleep(attemptInterval);
					}

					const auto logAttemptFailure = [&] {
						Logging::logWarning("Failed to perform operation (attempt %d" + (attempts > 0 ? "/" + Text::toString(attempts) : "") + ")", attempt);
					};

					try {
						Logging::logTrace("Attempting to perform operation (attempt %d" + (attempts > 0 ? "/" + Text::toString(attempts) : "") + ")", attempt);
						operation();

						return;
					} catch(const EnergyManager::Utility::Exceptions::Exception& exception) {
						exception.log();
						logAttemptFailure();
					} catch(const std::exception& exception) {
						EnergyManager::Utility::Exceptions::Exception(exception.what(), __FILE__, __LINE__).log();
						logAttemptFailure();
					} catch(...) {
						logAttemptFailure();
					}
				}

				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Failed to perform operation");
			}

			Exception::Exception(const std::string& message, std::string file, const size_t& line) : std::runtime_error(message), message_(message), file_(std::move(file)), line_(line) {
			}

			std::string Exception::getMessage() const {
				return message_;
			}

			std::string Exception::getFile() const {
				return file_;
			}

			size_t Exception::getLine() const {
				return line_;
			}

			void Exception::throwWithStacktrace() const {
				throw boost::enable_error_info(*this) << traced(boost::stacktrace::stacktrace());
			}

			void Exception::log() const {
				// Prepare the message
				std::string message = getMessage();

				// Collect the stacktrace, if any
				const boost::stacktrace::stacktrace* stacktrace = boost::get_error_info<traced>(*this);
				if(stacktrace) {
					message += "\n" + boost::stacktrace::to_string(*stacktrace);
				}

				// Show the error
				logError(message, getFile(), getLine());

				// Flush the buffers to ensure that everything is logged
				Logging::flush();
			}
		}
	}
}