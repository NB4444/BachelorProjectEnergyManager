/**************************************************************
*	Energy Aware Runtime (EAR)
*	This program is part of the Energy Aware Runtime (EAR).
*
*	EAR provides a dynamic, transparent and ligth-weigth solution for
*	Energy management.
*
*    	It has been developed in the context of the Barcelona Supercomputing Center (BSC)-Lenovo Collaboration project.
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
*	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*	The GNU LEsser General Public License is contained in the file COPYING
*/
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "frequency_intel63.h"

#define debug(...) \
	fprintf(stderr, __VA_ARGS__);

static uint cpu_count;
static topology_t tp;

typedef struct aperf_intel63_s
{
	ulong	freq_nominal;
	ulong	freq_aperf1;
	ulong	freq_aperf2;
	ulong	freq_mperf1;
	ulong	freq_mperf2;
	uint	snapped;
	uint	cpu_id;
	uint	error;
} aperf_intel63_t;

state_t freq_intel63_init(frequency_effective_t *ef, topology_t *_tp)
{
	size_t size = sizeof(aperf_intel63_t);
	aperf_intel63_t *a;
	state_t s;
	int cpu;

	// Static data
	if (cpu_count == 0) {
		cpu_count = _tp->core_count;
	}
	if (tp.cpus == NULL) {
		if(xtate_fail(s, topology_select(_tp, &tp, TPSelect.core, TPGroup.merge, 0))) {
			return s;
		}
	}

	// Allocating space
	if (posix_memalign((void **) &ef->data, 64, size * cpu_count) != 0) {
		return EAR_ERROR;
	}

	// Initialization
	a = (aperf_intel63_t *) ef->data;

	for (cpu = 0; cpu < cpu_count; ++cpu)
	{
		a[cpu].cpu_id		= tp.cpus[cpu].id;
		a[cpu].freq_nominal = tp.cpus[cpu].freq_base;

        if (xtate_fail(s, msr_open(a[cpu].cpu_id))) {
            return s;
        }
	}
	return EAR_SUCCESS;
}

state_t freq_intel63_dispose(frequency_effective_t *ef)
{
    aperf_intel63_t *a = ef->data;
	int cpu;

	for (cpu = 0; cpu < cpu_count; ++cpu) {
		msr_close(a[cpu].cpu_id);
	}

	free(a);
	
	return EAR_SUCCESS;
}

state_t freq_intel63_read(frequency_effective_t *ef, ulong *freq)
{
	ulong mperf_diff, aperf_diff, aperf_pcnt;
    aperf_intel63_t *a = ef->data;
	state_t result1, result2;
	int cpu;

	for (cpu = 0; cpu < cpu_count; ++cpu)
	{
		//
		a[cpu].freq_mperf1 = a[cpu].freq_mperf2;
		a[cpu].freq_aperf1 = a[cpu].freq_aperf2;
		//
		result1 = msr_read(a[cpu].cpu_id, &a[cpu].freq_mperf2, sizeof(ulong), MSR_IA32_MPERF);
		result2 = msr_read(a[cpu].cpu_id, &a[cpu].freq_aperf2, sizeof(ulong), MSR_IA32_APERF);
		//
		mperf_diff = a[cpu].freq_mperf2 - a[cpu].freq_mperf1;
		aperf_diff = a[cpu].freq_aperf2 - a[cpu].freq_aperf1;
		//
		if (((ulong) (-1LU) / 100LU) < aperf_diff)
		{
			aperf_diff >>= 7;
			mperf_diff >>= 7;
		}
		aperf_pcnt = (aperf_diff * 100LU) / mperf_diff;
		//
		a[cpu].snapped = !(result1 != EAR_SUCCESS || result2 != EAR_SUCCESS);
		a[cpu].error   = !(a[cpu].snapped);
		//
		freq[cpu] = (a[cpu].freq_nominal * aperf_pcnt) / 100LU;
		
		if (a[cpu].error) {
			freq[cpu] = 0LU;
		}
	}

	return EAR_SUCCESS;
}

state_t freq_intel63_read_count(frequency_effective_t *ef, uint *count)
{
	*count = cpu_count;	
	return EAR_SUCCESS;
}
