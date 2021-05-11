#include <cstdio>
//#include <cuda_runtime_api.h>
//#include <cuda.h>
#include <cupti.h>
#include <cupti_callbacks.h>
//#include <dlfcn.h>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <map>
#include <sys/stat.h>
#include <unistd.h>

#define DRIVER_API_CALL(apiFuncCall) \
	do { \
		CUresult _status = apiFuncCall; \
		if(_status != CUDA_SUCCESS) { \
			fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", __FILE__, __LINE__, #apiFuncCall, _status); \
			exit(-1); \
		} \
	} while(0)

#define RUNTIME_API_CALL(apiFuncCall) \
	do { \
		cudaError_t _status = apiFuncCall; \
		if(_status != cudaSuccess) { \
			fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
			exit(-1); \
		} \
	} while(0)

#define CUPTI_CALL(call) \
	do { \
		CUptiResult _status = call; \
		if(_status != CUPTI_SUCCESS) { \
			const char* errstr; \
			cuptiGetResultString(_status, &errstr); \
			fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr); \
			exit(-1); \
		} \
	} while(0)

/**
 * Whether to report events.
 */
bool enableEvents = true;

/**
 * Whether to report metrics.
 */
bool enableMetrics = false;

/**
 * The callback subscriber.
 */
CUpti_SubscriberHandle subscriber;

/**
 * GPU to use.
 */
CUdevice device = 0;

/**
 * The metric to record.
 */
const char* metricName = "ipc";

/**
 * The time event tracking started.
 */
std::chrono::system_clock::time_point startTime;

/**
 * User data for event collection callback.
 */
struct MetricData {
	/**
	 * The device where metric is being collected.
	 */
	CUdevice device;

	/**
	 * The set of event groups to collect for a pass.
	 */
	CUpti_EventGroupSet* eventGroups;

	/**
	 * The current number of events collected in eventIdArray and eventValueArray.
	 */
	uint32_t eventIdx;

	/**
	 * The number of entries in eventIdArray and eventValueArray.
	 */
	uint32_t numEvents;

	/**
	 * Array of event IDs.
	 */
	CUpti_EventID* eventIdArray;

	/**
	 * Array of event values.
	 */
	uint64_t* eventValueArray;
};

/**
 * The metric data
 */
MetricData* metricData;

bool fileExists(const char* name) {
	struct stat buffer {};
	return stat(name, &buffer) == 0;
}

void processEvents(void* userData, CUpti_CallbackDomain domain, CUpti_CallbackId callbackID, const CUpti_CallbackData* callbackInformation) {
	//printf("Processing events...\n");

	// Define the file names
	const auto eventsFile = "reporter.events.csv.tmp";
	const auto lockFile = "reporter.events.csv.tmp.lock";

	// Try to obtain an atomic file lock
	while(open(lockFile, O_CREAT | O_EXCL) < 0) {
		usleep(10);
	}

	// Check if the file already existed, and if not, write the headers
	if(!fileExists(eventsFile)) {
		//printf("Writing Reporter events headers...\n");

		// Open file to write headers
		const auto output = fopen(eventsFile, "w");
		if(output == nullptr) {
			printf("Failed to open events file for writing with error code %d: %s\n", errno, strerror(errno));
		}

		// Write the headers
		fputs("Timestamp;Event;Site\n", output);

		// Close the stream
		fclose(output);
	}

	// Get information about the event
	const auto timestamp = std::chrono::system_clock::now();
	const auto functionName = callbackInformation->functionName;
	const auto callbackSite = callbackInformation->callbackSite;
	const auto returnCode = (ulong) callbackInformation->functionReturnValue & 0x00000fff;
	const auto parameters = (ulong) callbackInformation->functionParams & 0x00000fff;
	const auto processID = (ulong) getpid() & 0x00000fff;
	const auto thread = (ulong) pthread_self() & 0x0000ffff;

	//printf("Processing event %s...\n", functionName);

	// Generate the output message
	char message[1024];
	snprintf(
		message,
		sizeof(message),
		"%ld;%s;%s\n",
		std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count(),
		functionName,
		callbackSite == CUPTI_API_ENTER ? "ENTER" : "EXIT");

	// Open file to write headers
	const auto output = fopen(eventsFile, "a");
	if(output == nullptr) {
		printf("Failed to open file for writing with error code %d: %s\n", errno, strerror(errno));
	}

	// Write the headers
	fputs(message, output);

	// Close the stream
	fclose(output);

	// Remove the lock
	remove(lockFile);
}

void processMetrics(void* userData, CUpti_CallbackDomain domain, CUpti_CallbackId callbackID, const CUpti_CallbackData* callbackInformation) {
	//printf("Processing metrics...\n");

	// This callback is enabled only for launch so we shouldn't see anything else.
	if((callbackID != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) && (callbackID != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
		//printf("Unexpected event\n");
		return;
	} else {
		//printf("Processing event metrics...\n");
	}

	auto* metricData = (MetricData*) userData;

	// Keep track of event data between loops
	const auto now = std::chrono::system_clock::now();

	// Read and record event values of the last cycle
	//if(initialized) {
	if(callbackInformation->callbackSite == CUPTI_API_EXIT) {
		//printf("Collecting metrics...\n");

		cudaDeviceSynchronize();

		// For each group, read the event values from the group and record in metricData
		for(unsigned int i = 0; i < metricData->eventGroups->numEventGroups; i++) {
			CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
			CUpti_EventDomainID groupDomain;
			uint32_t numEvents;
			uint32_t numInstances;
			uint32_t numTotalInstances;
			CUpti_EventID* eventIds;
			size_t groupDomainSize = sizeof(groupDomain);
			size_t numEventsSize = sizeof(numEvents);
			size_t numInstancesSize = sizeof(numInstances);
			size_t numTotalInstancesSize = sizeof(numTotalInstances);
			uint64_t* values;
			uint64_t normalized;
			uint64_t sum;
			size_t valuesSize;
			size_t eventIdsSize;

			CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize, &groupDomain));
			CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(metricData->device, groupDomain, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &numTotalInstancesSize, &numTotalInstances));
			CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &numInstancesSize, &numInstances));
			CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &numEventsSize, &numEvents));
			eventIdsSize = numEvents * sizeof(CUpti_EventID);
			eventIds = (CUpti_EventID*) malloc(eventIdsSize);
			CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, eventIds));

			valuesSize = sizeof(uint64_t) * numInstances;
			values = (uint64_t*) malloc(valuesSize);

			for(unsigned int j = 0; j < numEvents; j++) {
				CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE, eventIds[j], &valuesSize, values));
				if(metricData->eventIdx >= metricData->numEvents) {
					printf("Too many events collected, metric expects only %d\n", (int) metricData->numEvents);
					return;
					//exit(-1);
				}

				// Sum collect event values from all instances
				sum = 0;
				for(unsigned int k = 0; k < numInstances; k++) {
					sum += values[k];
				}

				// Normalize the event value to represent the total number of domain instances on the device
				normalized = (sum * numTotalInstances) / numInstances;

				metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
				metricData->eventValueArray[metricData->eventIdx] = normalized;
				metricData->eventIdx++;

				//// Print collected value
				//{
				//	char eventName[128];
				//	size_t eventNameSize = sizeof(eventName) - 1;
				//	CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME, &eventNameSize, eventName));
				//	eventName[127] = '\0';
				//	printf("\t%s = %llu (", eventName, (unsigned long long) sum);
				//	if(numInstances > 1) {
				//		for(unsigned int k = 0; k < numInstances; k++) {
				//			if(k != 0)
				//				printf(", ");
				//			printf("%llu", (unsigned long long) values[k]);
				//		}
				//	}
				//
				//	printf(")\n");
				//	printf("\t%s (normalized) (%llu * %u) / %u = %llu\n", eventName, (unsigned long long) sum, numTotalInstances, numInstances, (unsigned long long) normalized);
				//}
			}

			free(values);
		}

		if(metricData->eventIdx != metricData->numEvents) {
			printf("Error: expected %u metric events, got %u\n", metricData->numEvents, metricData->eventIdx);
			return;
			//exit(-1);
		}

		// Use all the collected events to calculate the metric value
		CUpti_MetricID metricId;
		CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName, &metricId));
		CUpti_MetricValue metricValue;
		CUPTI_CALL(cuptiMetricGetValue(
			device,
			metricId,
			metricData->numEvents * sizeof(CUpti_EventID),
			metricData->eventIdArray,
			metricData->numEvents * sizeof(uint64_t),
			metricData->eventValueArray,
			std::chrono::duration_cast<std::chrono::nanoseconds>(now - startTime).count(),
			&metricValue));

		// Generate the output message
		const auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
		char message[1024];

		CUpti_MetricValueKind valueKind;
		size_t valueKindSize = sizeof(valueKind);
		CUPTI_CALL(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND, &valueKindSize, &valueKind));
		switch(valueKind) {
			case CUPTI_METRIC_VALUE_KIND_DOUBLE:
				snprintf(message, sizeof(message), "%ld;%s;%f\n", timestamp, metricName, metricValue.metricValueDouble);
				break;
			case CUPTI_METRIC_VALUE_KIND_UINT64:
				snprintf(message, sizeof(message), "%ld;%s;%llu\n", timestamp, metricName, (unsigned long long) metricValue.metricValueUint64);
				break;
			case CUPTI_METRIC_VALUE_KIND_INT64:
				snprintf(message, sizeof(message), "%ld;%s;%lld\n", timestamp, metricName, (long long) metricValue.metricValueInt64);
				break;
			case CUPTI_METRIC_VALUE_KIND_PERCENT:
				snprintf(message, sizeof(message), "%ld;%s;%f\n", timestamp, metricName, metricValue.metricValuePercent);
				break;
			case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
				snprintf(message, sizeof(message), "%ld;%s;%llu\n", timestamp, metricName, (unsigned long long) metricValue.metricValueThroughput);
				break;
			case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
				snprintf(message, sizeof(message), "%ld;%s;%u\n", timestamp, metricName, (unsigned int) metricValue.metricValueUtilizationLevel);
				break;
			default:
				printf("Error: unknown value kind\n");
				return;
				//exit(-1);
		}

		{
			// Define the file names
			const auto metricsFile = "reporter.metrics.csv.tmp";
			const auto lockFile = "reporter.metrics.csv.tmp.lock";

			// Try to obtain an atomic file lock
			while(open(lockFile, O_CREAT | O_EXCL) < 0) {
				usleep(10);
			}

			// Check if the file already existed, and if not, write the headers
			if(!fileExists(metricsFile)) {
				//printf("Writing Reporter metrics headers...\n");

				// Open file to write headers
				const auto output = fopen(metricsFile, "w");
				if(output == nullptr) {
					printf("Failed to open metrics file for writing with error code %d: %s\n", errno, strerror(errno));
				}

				// Write the headers
				fputs("Timestamp;Metric;Value\n", output);

				// Close the stream
				fclose(output);
			}

			// Open file to write message
			const auto output = fopen(metricsFile, "a");
			if(output == nullptr) {
				printf("Failed to open file for writing with error code %d: %s\n", errno, strerror(errno));
			}

			// Write the message
			fputs(message, output);

			// Close the stream
			fclose(output);

			// Remove the lock
			remove(lockFile);
		}

		for(unsigned int i = 0; i < metricData->eventGroups->numEventGroups; i++) {
			CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
		}
	}

	// Enable all the event groups being collected this pass, for metrics we collect for all instances of the event
	//if(!initialized) {
	if(callbackInformation->callbackSite == CUPTI_API_ENTER) {
		cudaDeviceSynchronize();

		//printf("Initializing metrics...\n");
		metricData->eventIdx = 0;

		//printf("Configuring passes...\n");
		CUpti_MetricID metricId;
		CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName, &metricId));
		// Get the number of passes required to collect all the events needed for the metric and the event groups for each pass
		CUpti_EventGroupSets* passData;
		CUPTI_CALL(cuptiMetricCreateEventGroupSets(callbackInformation->context, sizeof(metricId), &metricId, &passData));
		//CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(metricId), &metricId, &passData));
		if(passData->numSets == 0) {
			printf("No sets generated\n");
			exit(1);
		} else if(passData->numSets > 1) {
			printf("Cannot initialize reader as profiling would require multiple passes\n");
			exit(1);
		} else {
			//printf("Initializing event groups...\n");
			metricData->eventGroups = passData->sets;

			//printf("Event groups initialized: %d\n", passData->sets->numEventGroups);
		}

		//printf("Configuring metrics...\n");
		CUPTI_CALL(cuptiSetEventCollectionMode(callbackInformation->context, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));

		//printf("Processing %d event groups...\n", metricData->eventGroups->numEventGroups);
		for(unsigned int i = 0; i < metricData->eventGroups->numEventGroups; i++) {
			const auto eventGroup = metricData->eventGroups->eventGroups[i];

			uint32_t all = 1;
			CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all));
			CUPTI_CALL(cuptiEventGroupEnable(eventGroup));
		}

		//printf("Marking device as initialized...\n");

		// Mark the device as initialized
		startTime = now;

		//printf("Metric collection initialized\n");
	}
}

void CUPTIAPI cuptiEventCallback(void* userData, CUpti_CallbackDomain domain, CUpti_CallbackId callbackID, const CUpti_CallbackData* callbackInformation) {
	if(enableEvents) {
		processEvents(userData, domain, callbackID, callbackInformation);
	}

	if(enableMetrics) {
		processMetrics(userData, domain, callbackID, callbackInformation);
	}
}

void __attribute__((constructor)) constructCUDAModule() {
	printf("Initializing Reporter...\n");

	// Remove existing files
	remove("reporter.events.csv.tmp");
	remove("reporter.events.csv.tmp.lock");
	remove("reporter.metrics.csv.tmp");
	remove("reporter.metrics.csv.tmp.lock");

	printf("Initializing CUPTI...\n");

	DRIVER_API_CALL(cuInit(0));
	int deviceCount;
	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if(deviceCount == 0) {
		printf("There is no device supporting CUDA.\n");
		exit(-2);
	}

	printf("Determining current device...\n");

	int deviceNum = 0;
	printf("CUDA Device Number: %d\n", deviceNum);

	DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
	char deviceName[32];
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
	printf("CUDA Device Name: %s\n", deviceName);

	if(enableMetrics) {
		CUcontext context = 0;
		DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

		printf("Allocating metric data for metric %s...\n", metricName);

		// Allocate space to hold all the events needed for the metric
		metricData = new MetricData();
		CUpti_MetricID metricId;
		CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName, &metricId));
		CUPTI_CALL(cuptiMetricGetNumEvents(metricId, &metricData->numEvents));
		metricData->device = device;
		metricData->eventIdArray = (CUpti_EventID*) malloc(metricData->numEvents * sizeof(CUpti_EventID));
		metricData->eventValueArray = (uint64_t*) malloc(metricData->numEvents * sizeof(uint64_t));
		metricData->eventIdx = 0;
	}

	printf("Initializing data collection...\n");

	// Subscribe to events
	cuptiSubscribe(&subscriber, (CUpti_CallbackFunc) cuptiEventCallback, metricData);
	cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
	cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
	if(enableMetrics) {
		CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
		CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
	}

	printf("Reporter initialized\n");
}

void __attribute__((destructor)) destructCUDAModule() {
	// Clean up metric data
	delete metricData;
}
