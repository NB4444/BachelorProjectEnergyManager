#include "./Exception.hpp"

#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Application.hpp"

#include <unistd.h>
#include <sys/wait.h>
// Get REG_EIP from ucontext.h
#define __USE_GNU
#include <signal.h>
#include <ucontext.h>

namespace EnergyManager {
	namespace Utility {
		namespace Exceptions {
			void Exception::backtraceSignalHandler(int signal, siginfo_t* info, void* secret) {
				// Do something useful with siginfo_t
				ucontext_t* ucontext = static_cast<ucontext_t*>(secret);
				if(signal == SIGSEGV) {
					char address[256];
					sprintf(address, "%p", info->si_addr);
					char fromAdrress[256];
					sprintf(fromAdrress, "%p", (void*)ucontext->uc_mcontext.gregs[REG_RIP]);

					Exception("Got signal " + std::to_string(signal) + ", faulty address is " + std::string(address) + ", from " + std::string(fromAdrress), __FILE__, __LINE__).log();
				} else {
					Exception("Got signal " + std::to_string(signal), __FILE__, __LINE__).log();
				}

				exit(0);
			}

			void Exception::initialize() {
				// Install the signal handler
				struct sigaction signalAction;
				signalAction.sa_sigaction = backtraceSignalHandler;
				sigemptyset(&signalAction.sa_mask);
				signalAction.sa_flags = SA_RESTART | SA_SIGINFO;

				sigaction(SIGSEGV, &signalAction, nullptr);
				sigaction(SIGTERM, &signalAction, nullptr);
				sigaction(SIGUSR1, &signalAction, nullptr);
			}

			void Exception::logGDBStackTrace() {
				// Create a buffer for process IDs
				char processIDBuffer[30];
				sprintf(processIDBuffer, "%d", getpid());

				// Create a buffer for process names
				char nameBuffer[1024];
				nameBuffer[readlink("/proc/self/exe", nameBuffer, sizeof(nameBuffer) - 1)] = 0;

				// Start a GDB instance as child process
				int childProcessID = fork();
				if(!childProcessID) {
					// Redirect output to stderr
					dup2(2, 1);
					fprintf(stdout, "Stack trace for %s pid=%s\n", nameBuffer, processIDBuffer);
					execlp("gdb", "gdb", "--batch", "-n", "-ex", "thread", "-ex", "bt", nameBuffer, processIDBuffer, nullptr);

					// Stop if GDB failed to run
					abort();
				} else {
					waitpid(childProcessID, nullptr, 0);
				}
			}

			Exception::Exception(const std::string& message, const std::string& file, const size_t& line) : std::runtime_error(message), file_(file), line_(line) {
				// Get the current stack trace
				void* traces[16];
				int traceSize = backtrace(traces, sizeof(traces));
				char** messages = backtrace_symbols(traces, traceSize);

				// Skip first stack frame (points here)
				for(int index = 1; index < traceSize; ++index) {
					// Find first occurence of '(' or ' ' in message[i] and assume everything before that is the file name
					// Don't go beyond 0 though (string terminator)
					size_t endIndex = 0;
					while(messages[index][endIndex] != '(' /*&& messages[i][index] != ' '*/ && messages[index][endIndex] != 0) {
						++endIndex;
					}

					// Get the current trace properties
					std::string message = messages[index];
					char address[256];
					sprintf(address, "%p", traces[index]);
					auto executable = message.substr(0, endIndex);

					// Extract the line number
					Application application("addr2line");
					application.run({ "-e \"" + executable + "\"", address });

					// Add the trace
					stackTrace_.emplace_back(messages[index], Text::trim(application.getExecutableOutput()));
				}
			}

			std::string Exception::getMessage() const {
				return what();
			}

			std::string Exception::getFile() const {
				return file_;
			}

			size_t Exception::getLine() const {
				return line_;
			}

			void Exception::log() const {
				Logging::logError(getMessage(), getFile(), getLine());

				Logging::logError("Stack trace:", getFile(), getLine());
				for(const auto& stackTrace : stackTrace_) {
					auto message = stackTrace.first;
					auto line = stackTrace.second;
					Logging::logError("%s (line: %s)", getFile(), getLine(), message.c_str(), line.c_str());
				}
			}
		}
	}
}