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
	experiment<Particlefilter_floatProfiler>(arguments, 50); // 6s * 3
	
	return 0;
}