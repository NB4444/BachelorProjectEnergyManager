#include <cstdio>
//#include <cuda_runtime_api.h>
//#include <cuda.h>
#include <cupti.h>
#include <cupti_callbacks.h>
//#include <dlfcn.h>
#include <cerrno>
#include <chrono>
#include <map>
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

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) (((uintptr_t)(buffer) & ((align) -1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align) -1))) : (buffer))

//uint cupti_calls[1024];

CUpti_SubscriberHandle subscriber;

/**
 * Whether to log events.
 */
bool logEvents = false;

/**
 * GPU to use.
 */
CUdevice device = 0;

/**
 * The metric to record.
 */
const char* metricName = "ipc";

void processEvents(void* userData, CUpti_CallbackDomain domain, CUpti_CallbackId callbackID, const CUpti_CallbackData* callbackInformation) {
	if(logEvents) {
		printf("Processing events...\n");
	}

	//char* site = "exits";
	//if(cbInfo->callbackSite == CUPTI_API_ENTER) {
	//	site = "enter";
	//}
	//
	//cupti_calls[cbid] += cbInfo->callbackSite == CUPTI_API_ENTER;
	//ulong fret = (ulong) cbInfo->functionReturnValue & 0x00000fff;
	//ulong fpar = (ulong) cbInfo->functionParams & 0x00000fff;
	//
	//ulong proc = (ulong) getpid() & 0x00000fff;
	//ulong thrd = (ulong) pthread_self() & 0x0000ffff;
	//
	//if(cbInfo->callbackSite != CUPTI_API_ENTER) {
	//	// tprintf("%s||%lu||%lu||%llu||%llu||%s||%u||%lu||%lu",
	//	fprintf(stderr, "%s;%lu;%lu;%s;%u;%lu;%lu\n", cbInfo->functionName, proc, thrd, site, cupti_calls[cbid], fret, fpar);
	//}

	// Get information about the event
	const auto timestamp = std::chrono::system_clock::now();
	const auto functionName = callbackInformation->functionName;
	const auto callbackSite = callbackInformation->callbackSite;
	const auto returnCode = (ulong) callbackInformation->functionReturnValue & 0x00000fff;
	const auto parameters = (ulong) callbackInformation->functionParams & 0x00000fff;
	const auto processID = (ulong) getpid() & 0x00000fff;
	const auto thread = (ulong) pthread_self() & 0x0000ffff;

	// Only process function enter events
	if(logEvents) {
		printf("Processing event %s...\n", functionName);
	}

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
	const auto output = fopen("reporter.events.csv.tmp", "a");
	if(output == nullptr) {
		printf("Failed to open file for writing with error code %d: %s\n", errno, strerror(errno));
	}

	// Write the headers
	fputs(message, output);

	// Close the stream
	fclose(output);
}

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

void processMetrics(void* userData, CUpti_CallbackDomain domain, CUpti_CallbackId callbackID, const CUpti_CallbackData* callbackInformation) {
	printf("Processing metrics...\n");

	// This callback is enabled only for launch so we shouldn't see
	// anything else.
	if((callbackID != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) && (callbackID != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
		//return;
	}

	MetricData* metricData = (MetricData*) userData;

	// Keep track of event data between loops
	static bool initialized = false;
	static std::chrono::system_clock::time_point startTime;

	// Keep track of data for the current loop
	const auto now = std::chrono::system_clock::now();

	// Read and record event values of the last cycle
	if(initialized) {
		printf("Collecting metrics...\n");

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
					fprintf(stderr, "error: too many events collected, metric expects only %d\n", (int) metricData->numEvents);
					exit(-1);
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

				// Print collected value
				{
					char eventName[128];
					size_t eventNameSize = sizeof(eventName) - 1;
					CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME, &eventNameSize, eventName));
					eventName[127] = '\0';
					printf("\t%s = %llu (", eventName, (unsigned long long) sum);
					if(numInstances > 1) {
						for(unsigned int k = 0; k < numInstances; k++) {
							if(k != 0)
								printf(", ");
							printf("%llu", (unsigned long long) values[k]);
						}
					}

					printf(")\n");
					printf("\t%s (normalized) (%llu * %u) / %u = %llu\n", eventName, (unsigned long long) sum, numTotalInstances, numInstances, (unsigned long long) normalized);
				}
			}

			free(values);
		}

		if(metricData->eventIdx != metricData->numEvents) {
			fprintf(stderr, "error: expected %u metric events, got %u\n", metricData->numEvents, metricData->eventIdx);
			exit(-1);
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
				fprintf(stderr, "error: unknown value kind\n");
				exit(-1);
		}

		{
			// Open file to write headers
			const auto output = fopen("reporter.metrics.csv.tmp", "a");
			if(output == nullptr) {
				printf("Failed to open file for writing with error code %d: %s\n", errno, strerror(errno));
			}

			// Write the headers
			fputs(message, output);

			// Close the stream
			fclose(output);
		}

		for(unsigned int i = 0; i < metricData->eventGroups->numEventGroups; i++) {
			CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
		}
	}

	// Enable all the event groups being collected this pass, for metrics we collect for all instances of the event
	{
		printf("Configuring metrics...\n");
		CUPTI_CALL(cuptiSetEventCollectionMode(callbackInformation->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));
		printf("Processing %d event groups...\n", metricData->eventGroups->numEventGroups);
		for(unsigned int i = 0; i < metricData->eventGroups->numEventGroups; i++) {
			const auto eventGroup = metricData->eventGroups->eventGroups[i];

			uint32_t all = 1;
			CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all));
			CUPTI_CALL(cuptiEventGroupEnable(eventGroup));
			return;
		}

		printf("Marking device as initialized...\n");

		// Mark the device as initialized
		initialized = true;
		startTime = now;

		printf("Metric collection initialized\n");
	}
}

void CUPTIAPI cuptiEventCallback(void* userData, CUpti_CallbackDomain domain, CUpti_CallbackId callbackID, const CUpti_CallbackData* callbackInformation) {
	processEvents(userData, domain, callbackID, callbackInformation);
	//processMetrics(userData, domain, callbackID, callbackInformation);
}

void __attribute__((constructor)) constructCUDAModule() {
	printf("Initializing Reporter...\n");

	printf("Writing Reporter headers...\n");

	{
		// Open file to write headers
		const auto output = fopen("reporter.events.csv.tmp", "w");
		if(output == nullptr) {
			printf("Failed to open events file for writing with error code %d: %s\n", errno, strerror(errno));
		}

		// Write the headers
		fputs("Timestamp;Event;Site\n", output);

		// Close the stream
		fclose(output);
	}
	{
		// Open file to write headers
		const auto output = fopen("reporter.metrics.csv.tmp", "w");
		if(output == nullptr) {
			printf("Failed to open metrics file for writing with error code %d: %s\n", errno, strerror(errno));
		}

		// Write the headers
		fputs("Timestamp;Metric;Value\n", output);

		// Close the stream
		fclose(output);
	}

	printf("Initializing CUPTI...\n");

	DRIVER_API_CALL(cuInit(0));
	int deviceCount;
	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if(deviceCount == 0) {
		printf("There is no device supporting CUDA.\n");
		exit(-2);
	}

	printf("Determining current device...\n");

	int deviceNum;
	DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
	char deviceName[32];
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
	printf("CUDA Device Name: %s\n", deviceName);

	CUcontext context = 0;
	DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

	printf("Initializing metric collection...\n");

	// Subscribe to events
	MetricData metricData {};
	cuptiSubscribe(&subscriber, (CUpti_CallbackFunc) cuptiEventCallback, &metricData);
	cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
	//cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
	//CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
	//CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

	printf("Allocating metric data...\n");

	// Allocate space to hold all the events needed for the metric
	CUpti_MetricID metricId;
	CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName, &metricId));
	CUPTI_CALL(cuptiMetricGetNumEvents(metricId, &metricData.numEvents));
	metricData.eventIdArray = (CUpti_EventID*) malloc(metricData.numEvents * sizeof(CUpti_EventID));
	metricData.eventValueArray = (uint64_t*) malloc(metricData.numEvents * sizeof(uint64_t));
	metricData.eventIdx = 0;

	printf("Configuring passes...\n");

	// Get the number of passes required to collect all the events needed for the metric and the event groups for each pass
	CUpti_EventGroupSets* passData;
	printf("Metric ID: %d\n", metricId);
	//CUPTI_CALL(cuptiMetricCreateEventGroupSets(callbackInformation->context, sizeof(metricId), &metricId, &passData));
	CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(metricId), &metricId, &passData));
	for(unsigned int pass = 0; pass < passData->numSets; pass++) {
		metricData.eventGroups = passData->sets + pass;
	}
	if(passData->numSets > 1) {
		fprintf(stderr, "Cannot initialize reader as profiling would require multiple passes");
		exit(1);
	}

	printf("Reporter initialized\n");
}
