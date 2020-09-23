#pragma once

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			class SyntheticOperation {
			protected:
				virtual void onRun() = 0;

			public:
				void run();
			};
		}
	}
}