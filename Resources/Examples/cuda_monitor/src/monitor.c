/**************************************************************
 *	Energy Aware Runtime (EAR)
 *	This program is part of the Energy Aware Runtime (EAR).
 *
 *	EAR provides a dynamic, transparent and ligth-weigth solution for
 *	Energy managem	ent.
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

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "apis_suscriptor.h"
#include "common/pipes.h"
#include "common/string_enhanced.h"
#include "energy_gpu.h"
#include "metrics/frequency_intel63.h"

#define test(f) \
	if(state_fail(s)) { \
		printf("%s: %d (%s)\n", f, s, state_msg); \
		exit(-1); \
	}

int main(int argc, char* argv[]) {
	frequency_effective_t data_fe;
	gpu_energy_t* data_aux1;
	gpu_energy_t* data_aux2;
	topology_t data_tp1;
	topology_t data_tp2;
	api_ctx_t ctx_gpu;
	ulong freqs[1024];
	ulong timeout_ms;
	uint gpu_count;
	uint cpu_count;
	uint message;
	pipe_t pipes;
	uint burst;
	uint procs;
	state_t s;
	int i;
	int j;

	//
	api_clean(ctx_gpu);
	// GPU
	s = energy_gpu_init(&ctx_gpu, Monitor.suscription(), 1000);
	test("energy_gpu_init");
	s = energy_gpu_data_alloc(&ctx_gpu, &data_aux1);
	test("energy_gpu_data_alloc");
	s = energy_gpu_data_alloc(&ctx_gpu, &data_aux2);
	test("energy_gpu_data_alloc");
	s = energy_gpu_count(&ctx_gpu, &gpu_count);
	test("energy_gpu_count");
	// CPU Topology
	s = topology_init(&data_tp1);
	test("topology_init");
	// CPU Effective frequency
	s = freq_intel63_init(&data_fe, &data_tp1);
	test("freq_intel63_init");
	s = freq_intel63_read_count(&data_fe, &cpu_count);
	test("freq_intel63_read_count");
	s = topology_select(&data_tp1, &data_tp2, TPSelect.core, TPGroup.merge, 0);
	test("topology_select");
	// Suscriptor
	s = suscriptor_init();
	test("suscriptor_init");
	// Pipes
	s = pipes_open(&pipes, "./gpus.pipe", PIPES_RW, 1);
	test("pipes_init");

	//
	timeout_ms = 1000LU;
	timestamp_t t2;
	ulong ts;

	tprintf_init(fderr, STR_MODE_DEF, "5 12 8 8 7 6 5 5 5 4 4 4 4 4");

	while(1) {
		s = energy_gpu_read(&ctx_gpu, data_aux1);
		test("energy_gpu_read");

		// Select
		s = pipes_select(&pipes, timeout_ms);
		test("pipes_select");

		if(pipes_ready(pipes)) {
			s = pipes_read(&pipes, &message, sizeof(uint));
			test("pipes_read");

			printf("MESSAGE: GPUs are running (%d)\n", message);
			s = suscriptor_burst(ctx_gpu.suscription, 100);
			test("suscriptor_burst");
			timeout_ms = 200LU;
			burst = message;
		}

		s = energy_gpu_read(&ctx_gpu, data_aux2);
		test("energy_gpu_read");
		s = energy_gpu_data_diff(&ctx_gpu, data_aux2, data_aux1);
		test("energy_gpu_data_diff");

		// Counting GPU processes
		procs = 0;

		for(i = 0; i < gpu_count; ++i) {
			if(data_aux2[i].samples > 0) {
				ts = timestamp_getfast_convert(&t2, TIME_MSECS);

				// tprintf("gpu%u||%lu||%0.2lf||%0.2lf||%lu||%lu||%2lu%%||%lu%%||%lu||%lu||%lu||%lu||%lu||%lu",
				fprintf(
					stderr,
					"gpu%u;%lu;%0.2lf;%0.2lf;%lu;%lu;%lu;%lu;%lu;%lu;%lu;%lu;%lu;%lu\n",
					i,
					ts,
					data_aux2[i].energy_j,
					data_aux2[i].power_w,
					data_aux2[i].freq_gpu_mhz,
					data_aux2[i].freq_mem_mhz,
					data_aux2[i].util_gpu,
					data_aux2[i].util_mem,
					data_aux2[i].temp_gpu_cls,
					data_aux2[i].temp_mem_cls,
					data_aux2[i].procs_new_n,
					data_aux2[i].procs_cur_n,
					data_aux2[i].procs_tot_n,
					data_aux2[i].samples);

				procs += data_aux2[i].procs_cur_n;
			}
		}

		if(procs == 0 && burst == 1) {
			s = suscriptor_relax(ctx_gpu.suscription);
			test("suscriptor_relax");
			timeout_ms = 1000LU;
			burst = 0;
		} else if(procs > 0 && burst == 0) {
			s = suscriptor_burst(ctx_gpu.suscription, 100);
			test("suscriptor_burst");
			timeout_ms = 200LU;
			burst = 1;
		}

		// Frequencies
		s = freq_intel63_read(&data_fe, freqs);
		test("freq_intel63_read");

		for(j = 0; j < data_tp2.socket_count; ++j) {
			fprintf(stderr, "soc%d;%lu", j, ts);
			for(i = 0; i < cpu_count; ++i) {
				fprintf(stderr, ";%1.1lf", ((double) freqs[i]) / 1000000.0);
			}
			fprintf(stderr, "\n");
		}
	}

	return s;
}
