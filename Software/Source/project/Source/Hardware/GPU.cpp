#include "GPU.hpp"

namespace Hardware {
	void GPU::cuptiCall(const std::function<CUptiResult()>& call) const {
		do {
			CUptiResult callResult = call();
			if(callResult != CUPTI_SUCCESS) {
				const char* errorMessage;
				cuptiGetResultString(callResult, &errorMessage);
				fprintf(stderr, "%s:%d: error: CUPTI call failed with error %s.\n", __FILE__, __LINE__, errorMessage);
				if(callResult == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) {
					exit(0);
				} else {
					exit(-1);
				}
			}
		} while(0);
	}

	GPU::GPU() {
		// Enable collection of various types of parameters
		cuptiCall([] {
			return cuptiActivityEnable(CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT);
		});
	}

	uint32_t GPU::getTemperature() const {
		size_t result = 0;
		size_t resultSize = sizeof(size_t);


	}
}