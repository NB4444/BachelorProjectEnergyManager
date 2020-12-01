/**************************************************************
 *	Energy Aware Runtime (EAR)
 *	This program is part of the Energy Aware Runtime (EAR).
 *
 *	EAR provides a dynamic, transparent and ligth-weigth solution for
 *	Energy management.
 *
 *    	It has been developed in the context of the Barcelona Supercomputing
 *Center (BSC)-Lenovo Collaboration project.
 *
 *       Copyright (C) 2017
 *	BSC Contact 	mailto:ear-support@bsc.es
 *	Lenovo contact 	mailto:hpchelp@lenovo.com
 *
 *	EAR is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	EAR is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *	Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with EAR; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301
 *USA The GNU LEsser General Public License is contained in the file COPYING
 */

#include "energy_gpu_nvml.h"
#include <nvml.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static struct error_s {
  char *init;
  char *init_not;
  char *null_context;
  char *null_data;
  char *gpus_not;
} Error = {.init = "context already initialized or not empty",
           .init_not = "context not initialized",
           .null_context = "context pointer is NULL",
           .null_data = "data pointer is NULL",
           .gpus_not = "no GPUs detected"};

static pthread_mutex_t app_lock = PTHREAD_MUTEX_INITIALIZER;
typedef struct nvsml_s {
  nvmlProcessInfo_t *procs;
  gpu_energy_t *pool;
  uint num_gpus;
} nvsml_t;

#define nvml(c) ((nvsml_t *)c->context)

state_t nvml_status() {
  nvmlReturn_t r;
  uint n_gpus;

  if ((r = nvmlInit()) != NVML_SUCCESS) {
    return_msg(EAR_ERROR, nvmlErrorString(r));
  }
  if ((r = nvmlDeviceGetCount_v2(&n_gpus)) != NVML_SUCCESS) {
    return_msg(EAR_ERROR, nvmlErrorString(r));
  }
  if (n_gpus == 0) {
    return_msg(EAR_ERROR, Error.gpus_not);
  }

  nvmlShutdown();
  return EAR_SUCCESS;
}

state_t nvml_init(pcontext_t *c) {
  nvmlReturn_t r;
  nvsml_t *n;
  state_t s;

  if (c->initialized == 1) {
    return_msg(EAR_INITIALIZED, Error.init);
  }
  if ((r = nvmlInit()) != NVML_SUCCESS) {
    return_msg(EAR_ERROR, nvmlErrorString(r));
  }

  c->initialized = 1;
  c->context = calloc(1, sizeof(nvsml_t));
  n = (nvsml_t *)c->context;

  if (n == NULL) {
    nvml_dispose(c);
    return_msg(EAR_SYSCALL_ERROR, strerror(errno));
  }
  if (xtate_fail(s, nvml_count(c, &n->num_gpus))) {
    nvml_dispose(c);
    return s;
  }
  if (xtate_fail(s, nvml_data_alloc(c, &n->pool))) {
    nvml_dispose(c);
    return s;
  }
  n->procs = calloc(n->num_gpus * 2 * GPU_MAX_PROCS, sizeof(nvmlProcessInfo_t));
  if (n->procs == NULL) {
    return_msg(EAR_SYSCALL_ERROR, strerror(errno));
  }
#if 0
	int i;
	for (i = 0; i < n->num_gpus; ++i) {
		timestamp_getfast(&n->pool[i].time);
	}
#endif

  return EAR_SUCCESS;
}

state_t nvml_dispose(pcontext_t *c) {
  if (c->initialized != 1) {
    return_msg(EAR_NOT_INITIALIZED, Error.init_not);
  }
  if (c->context != NULL) {
    free(c->context);
  }

  c->initialized = 1;
  c->context = NULL;
  nvmlShutdown();

  return EAR_SUCCESS;
}

state_t nvml_count(pcontext_t *c, uint *count) {
  nvmlReturn_t r;

  if (c->initialized != 1) {
    return_msg(EAR_NOT_INITIALIZED, Error.init_not);
  }
  if (c->context == NULL) {
    return_msg(EAR_ERROR, Error.null_context);
  }
  if ((r = nvmlDeviceGetCount(count)) != NVML_SUCCESS) {
    *count = 0;
    return_msg(EAR_ERROR, nvmlErrorString(r));
  }
  if (((int)*count) <= 0) {
    *count = 0;
    return_msg(EAR_ERROR, Error.gpus_not);
  }
  return EAR_SUCCESS;
}

state_t nvml_read(pcontext_t *c, gpu_energy_t *data_read) {
  nvmlReturn_t s0, s1, s2, s3, s4, s5;
  nvsml_t *n = c->context;
  nvmlUtilization_t util;
  nvmlEnableState_t mode;
  nvmlDevice_t device;
  timestamp_t time;
  uint freq_gpu_mhz;
  uint freq_mem_mhz;
  uint temp_gpu;
  uint power_mw;

  if (c->initialized != 1) {
    return_msg(EAR_NOT_INITIALIZED, Error.init_not);
  }
  if (c->context == NULL) {
    return_msg(EAR_ERROR, Error.null_context);
  }

  //
  nvmlProcessInfo_t *pcur;
  nvmlProcessInfo_t *ppre;
  uint procs_n;
  uint procs_d;
  int i;
  int k;
  int j;

  timestamp_getfast(&time);

  while (pthread_mutex_trylock(&app_lock))
    ;

  for (i = 0; i < n->num_gpus; ++i) {
#define OK NVML_SUCCESS

    // Cleaning
    memset(&data_read[i], 0, sizeof(gpu_energy_t));

    // Testing if all is right
    if ((s0 = nvmlDeviceGetHandleByIndex(i, &device)) != OK) {
      continue;
    }
    if ((s0 = nvmlDeviceGetPowerManagementMode(device, &mode)) != OK) {
      continue;
    }
    if (mode != NVML_FEATURE_ENABLED) {
      continue;
    }

    // Data gathering definitions
    pcur = &n->procs[i * 2 * GPU_MAX_PROCS + GPU_MAX_PROCS];
    ppre = &n->procs[i * 2 * GPU_MAX_PROCS];
    procs_n = GPU_MAX_PROCS;
    procs_d = 0;

    // Getting the metrics by calling NVML (no MEM temp)
    s0 = nvmlDeviceGetPowerUsage(device, &power_mw);
    s1 = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &freq_mem_mhz);
    s2 = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &freq_gpu_mhz);
    s3 = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_gpu);
    s4 = nvmlDeviceGetUtilizationRates(device, &util);
    s5 = nvmlDeviceGetComputeRunningProcesses(device, &procs_n, pcur);

    // Pooling process differences O(n^2)
    for (k = 0; k < procs_n; ++k) {
      for (j = 0; j < GPU_MAX_PROCS; ++j) {
        if (pcur[k].pid == ppre[j].pid || ppre[j].pid == 0) {
          break;
        }
      }
      procs_d += (ppre[j].pid == 0 || j == GPU_MAX_PROCS);
    }
    for (k = 0; k < procs_n; ++k) {
      ppre[k].pid = pcur[k].pid;
    }
    for (; k < GPU_MAX_PROCS; ++k) {
      ppre[k].pid = 0;
    }

    // Pooling the data
    n->pool[i].power_w += (double)power_mw;
    n->pool[i].freq_mem_mhz += (ulong)freq_mem_mhz;
    n->pool[i].freq_gpu_mhz += (ulong)freq_gpu_mhz;
    n->pool[i].util_mem += (ulong)util.memory;
    n->pool[i].util_gpu += (ulong)util.gpu;
    n->pool[i].temp_gpu_cls += (ulong)temp_gpu;
    n->pool[i].temp_mem_cls += 0;
    n->pool[i].procs_tot_n += procs_d;
    n->pool[i].samples += 1;

    // Counting in the data
    data_read[i].samples = n->pool[i].samples;
    data_read[i].time = time;
    data_read[i].freq_mem_mhz = n->pool[i].freq_mem_mhz;
    data_read[i].freq_gpu_mhz = n->pool[i].freq_gpu_mhz;
    data_read[i].temp_gpu_cls = n->pool[i].temp_gpu_cls;
    data_read[i].temp_mem_cls = 0;
    data_read[i].procs_cur_n = procs_n;
    data_read[i].procs_tot_n = n->pool[i].procs_tot_n;
    data_read[i].procs_new_n = 0;
    data_read[i].util_mem = n->pool[i].util_mem;
    data_read[i].util_gpu = n->pool[i].util_gpu;
    data_read[i].energy_j = 0;
    data_read[i].power_w = n->pool[i].power_w;
    data_read[i].correct = 1;

// Removing warnings
#define unused(x) (void)(x)
    unused(s1);
    unused(s2);
    unused(s3);
    unused(s4);
    unused(s5);
  }

  pthread_mutex_unlock(&app_lock);

  return EAR_SUCCESS;
}

state_t nvml_data_alloc(pcontext_t *c, gpu_energy_t **data_read) {
  nvsml_t *n = c->context;

  if (c->context == NULL) {
    return_msg(EAR_ERROR, Error.null_context);
  }
  if (c->initialized != 1) {
    return_msg(EAR_NOT_INITIALIZED, Error.init_not);
  }
  if (data_read == NULL) {
    return_msg(EAR_ERROR, Error.null_data);
  }
  *data_read = calloc(n->num_gpus, sizeof(gpu_energy_t));
  if (*data_read == NULL) {
    return_msg(EAR_SYSCALL_ERROR, strerror(errno));
  }

  return EAR_SUCCESS;
}

state_t nvml_data_free(pcontext_t *c, gpu_energy_t **data_read) {
  if (data_read != NULL) {
    free(*data_read);
  }
  return EAR_SUCCESS;
}

state_t nvml_data_null(pcontext_t *c, gpu_energy_t *data_read) {
  nvsml_t *n = c->context;
  if (c->initialized != 1) {
    return_msg(EAR_NOT_INITIALIZED, Error.init_not);
  }
  if (c->context == NULL) {
    return_msg(EAR_ERROR, Error.null_context);
  }
  if (data_read == NULL) {
    return_msg(EAR_ERROR, Error.null_data);
  }
  memset(data_read, 0, n->num_gpus * sizeof(gpu_energy_t));

  return EAR_SUCCESS;
}

static void nvml_read_diff(gpu_energy_t *data_read2, gpu_energy_t *data_read1,
                           int i) {
  gpu_energy_t *d2 = &data_read2[i];
  gpu_energy_t *d1 = &data_read1[i];
  ullong utime;
  double dtime;

  if (d2->correct != 1 || d1->correct != 1) {
    memset(d2, 0, sizeof(gpu_energy_t));
    return;
  }

#if 0
	printf("########### %llu %llu\n",
		timestamp_convert(&d2->time, TIME_MSECS),
		timestamp_convert(&d1->time, TIME_MSECS));
#endif

  // Computing time
  utime = timestamp_diff(&d2->time, &d1->time, TIME_USECS);
  dtime = ((double)utime) / 1000000.0;
  // Averages
  d2->samples = d2->samples - d1->samples;

  if (d2->samples == 0) {
    memset(d2, 0, sizeof(gpu_energy_t));
    return;
  }

  d2->procs_new_n = (d2->procs_tot_n - d1->procs_tot_n);
  d2->freq_gpu_mhz = (d2->freq_gpu_mhz - d1->freq_gpu_mhz) / d2->samples;
  d2->freq_mem_mhz = (d2->freq_mem_mhz - d1->freq_mem_mhz) / d2->samples;
  d2->util_gpu = (d2->util_gpu - d1->util_gpu) / d2->samples;
  d2->util_mem = (d2->util_mem - d1->util_mem) / d2->samples;
  d2->temp_gpu_cls = (d2->temp_gpu_cls - d1->temp_gpu_cls) / d2->samples;
  d2->temp_mem_cls = 0;
  d2->power_w = (d2->power_w - d1->power_w) / (d2->samples * 1000);
  d2->energy_j = (d2->power_w) * dtime;
}

state_t nvml_data_diff(pcontext_t *c, gpu_energy_t *data_read2,
                       gpu_energy_t *data_read1) {
  nvsml_t *n = c->context;
  int i;

  if (c->initialized != 1) {
    return_msg(EAR_NOT_INITIALIZED, Error.init_not);
  }
  if (c->context == NULL) {
    return_msg(EAR_ERROR, Error.null_context);
  }
  if (data_read1 == NULL || data_read2 == NULL) {
    return_msg(EAR_ERROR, Error.null_data);
  }
  for (i = 0; i < n->num_gpus; i++) {
    nvml_read_diff(data_read2, data_read1, i);
  }

  return EAR_SUCCESS;
}

state_t nvml_data_copy(pcontext_t *c, gpu_energy_t *data_dst,
                       gpu_energy_t *data_src) {
  while (pthread_mutex_trylock(&app_lock))
    ;
  memcpy(data_dst, data_src, nvml(c)->num_gpus * sizeof(gpu_energy_t));
  pthread_mutex_unlock(&app_lock);
  return EAR_SUCCESS;
}
