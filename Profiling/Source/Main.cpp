#include "BFS.hpp"
#include "Jacobi.hpp"
#include "KMeans.hpp"
#include "MatrixMultiply.hpp"
#include "Experiment.hpp"

#include <EnergyManager/Hardware/Core.hpp>

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);
	
	// Run the tests
	
	//experiment<EnergyManager::Profiling::Profilers::BFSProfiler>(arguments, 50); // 58s * 3
	//experiment<UnifiedMemoryPerfProfiler>(arguments, 5); // 453s * 3
	//experiment<NWProfiler>(arguments, 200); // 28s * 3
	//experiment<BandwidthProfiler>(arguments, 50); 47s * 3
	//experiment<EnergyManager::Profiling::Profilers::KMeansProfiler>(arguments, 50); // 39s * 3
	//experiment<LavaMDProfiler>(arguments, 50); // 20s * 3
	//experiment<MyocyteProfiler>(arguments, 50); // 42s * 3
	//experiment<SRAD_V1Profiler>(arguments, 50); // 8.5s * 3
	//experiment<StreamclusterProfiler>(arguments, 10); // 80s * 3
	//experiment<EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler>(arguments, 10); // 64s * 3
	//experiment<Particlefilter_floatProfiler>(arguments, 50); // 6s * 3
	//experiment<JacobiNoMPIProfiler>(arguments, 10); // unoptimized: 90 * 3 optimized: 80 * 3
	
	auto n = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "-n", 0);

	switch(n) {
		case 0:
			//experiment<StreamclusterProfiler>(arguments, 10);
			experiment<EnergyManager::Profiling::Profilers::BFSProfiler>(arguments, 50);
			//experiment<MyocyteProfiler>(arguments, 75);
			//experiment<LavaMDProfiler>(arguments, 150);
			//experiment<SRAD_V1Profiler>(arguments, 350);
			//experiment<NWProfiler>(arguments, 400);
			//experiment<Particlefilter_floatProfiler>(arguments, 500);
			//experiment<EnergyManager::Profiling::Profilers::KMeansProfiler>(arguments, 75);
			//experiment<BandwidthProfiler>(arguments, 50);
			experiment<UnifiedMemoryPerfProfiler>(arguments, 3);
			experiment<EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler>(arguments, 10);
			experiment<JacobiNoMPIProfiler>(arguments, 3);
			break;
		case 1:
			experiment<StreamclusterProfiler>(arguments, 10);
			break;
		case 2:
			experiment<EnergyManager::Profiling::Profilers::BFSProfiler>(arguments, 50);
			break;
		case 3:
			experiment<MyocyteProfiler>(arguments, 75);
			break;
		case 4:
			experiment<LavaMDProfiler>(arguments, 150);
			break;
		case 5:
			experiment<SRAD_V1Profiler>(arguments, 350);
			break;
		case 6:
			experiment<NWProfiler>(arguments, 400);
			break;
		case 7:
			experiment<Particlefilter_floatProfiler>(arguments, 500);
			break;
		case 8:
			experiment<EnergyManager::Profiling::Profilers::KMeansProfiler>(arguments, 75);
			break;
		case 9:
			experiment<BandwidthProfiler>(arguments, 50);
			break;
		case 10:
			experiment<UnifiedMemoryPerfProfiler>(arguments, 3);
			break;
		case 11:
			experiment<EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler>(arguments, 10);
			break;
		case 12:
			experiment<JacobiNoMPIProfiler>(arguments, 3);
			break;
		default:
			break;
	}
	
	return 0;
}