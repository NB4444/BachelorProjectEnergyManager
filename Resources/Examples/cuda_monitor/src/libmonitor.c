#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <cupti_callbacks.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>

#include "common/pipes.h"
#include "common/string_enhanced.h"
#include "common/time.h"

uint cupti_calls[1024];

CUpti_SubscriberHandle subscriber;
__thread timestamp_t t1;
__thread timestamp_t t2;
__thread timestamp_t t0;
__thread ullong ts;
__thread ullong to;

void CUPTIAPI cuda_callback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo) {
	char* site = "exits";
	if(cbInfo->callbackSite == CUPTI_API_ENTER) {
		site = "enter";
	}

	timestamp_getfast(&t2);
	to = timestamp_getfast_convert(&t2, TIME_MSECS);
	ts = timestamp_diff(&t2, &t1, TIME_MSECS);

	cupti_calls[cbid] += cbInfo->callbackSite == CUPTI_API_ENTER;
	ulong fret = (ulong) cbInfo->functionReturnValue & 0x00000fff;
	ulong fpar = (ulong) cbInfo->functionParams & 0x00000fff;

	ulong proc = (ulong) getpid() & 0x00000fff;
	ulong thrd = (ulong) pthread_self() & 0x0000ffff;

	if(cbInfo->callbackSite != CUPTI_API_ENTER) {
		// tprintf("%s||%lu||%lu||%llu||%llu||%s||%u||%lu||%lu",
		fprintf(stderr, "%s;%lu;%lu;%llu;%llu;%s;%u;%lu;%lu\n", cbInfo->functionName, proc, thrd, to, ts, site, cupti_calls[cbid], fret, fpar);
	}

	timestamp_getfast(&t1);
}

void __attribute__((constructor)) module_cuda() {
	tprintf_init(fderr, STR_MODE_DEF, "40 8 8 12 8 8 8 8 8");
	tprintf(
		"Function name||Proc||Thread||Timestamp||Time plus||Site||#||Ret "
		"val.||Ret par.");

	timestamp_getfast(&t1);
	cuptiSubscribe(&subscriber, (CUpti_CallbackFunc) cuda_callback, NULL);
	cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
	//    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
}
