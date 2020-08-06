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

#include <common/states.h>
#include <metrics/energy_gpu_nvml.h>
#include <energy_gpu.h>

struct energy_gpu_ops
{
	state_t (*init)			(pcontext_t *c);
	state_t (*dispose)		(pcontext_t *c);
	state_t (*count)		(pcontext_t *c, uint *count);
	state_t (*read)			(pcontext_t *c, gpu_energy_t  *dr);
	state_t (*data_alloc)	(pcontext_t *c, gpu_energy_t **dr);
	state_t (*data_free)	(pcontext_t *c, gpu_energy_t **dr);
	state_t (*data_null)	(pcontext_t *c, gpu_energy_t  *dr);
	state_t (*data_diff)	(pcontext_t *c, gpu_energy_t  *dr2, gpu_energy_t *dr1);
	state_t (*data_copy)	(pcontext_t *c, gpu_energy_t  *dst, gpu_energy_t *src);
} ops;

state_t energy_gpu_init(pcontext_t *c, suscription_t *s, uint loop_ms)
{
	state_t r;

	if (state_ok(nvml_status())) {
		ops.init		= nvml_init;
		ops.dispose		= nvml_dispose;
		ops.read		= nvml_read;
		ops.count		= nvml_count;
		ops.data_alloc	= nvml_data_alloc;
		ops.data_free	= nvml_data_free;
		ops.data_null	= nvml_data_null;
		ops.data_diff	= nvml_data_diff;
		ops.data_copy   = nvml_data_copy;
	} else {
		return_msg(EAR_ERROR, "no energy GPU API available");
	}

	if (xtate_fail(r, ops.init(c))) {
		return r;
	}

	if (s != NULL)
	{
		ops.data_alloc(c, (gpu_energy_t **) &s->api_memm);
		s->api_call = (void *) ops.read;
		s->api_time = loop_ms;
		s->api_cntx = c;
		return s->suscribe(s);
	}

	return EAR_SUCCESS;
}

state_t energy_gpu_dispose(pcontext_t *c)
{
	preturn(ops.dispose, c);
}

state_t energy_gpu_read(pcontext_t *c, gpu_energy_t *data_read)
{
	if (suscribed(c)) {
		return energy_gpu_data_copy(c, data_read, sus(c)->api_memm);
	}
	preturn(ops.read, c, data_read);
}

state_t energy_gpu_count(pcontext_t *c, uint *count)
{
	preturn(ops.count, c, count);
}

state_t energy_gpu_data_alloc(pcontext_t *c, gpu_energy_t **data_read)
{
	preturn(ops.data_alloc, c, data_read);
}

state_t energy_gpu_data_free(pcontext_t *c, gpu_energy_t **data_read)
{
	preturn(ops.data_free, c, data_read);
}

state_t energy_gpu_data_null(pcontext_t *c, gpu_energy_t *data_read)
{
	preturn(ops.data_null, c, data_read);
}

state_t energy_gpu_data_diff(pcontext_t *c, gpu_energy_t *data_read2, gpu_energy_t *data_read1)
{
	preturn(ops.data_diff, c, data_read2, data_read1);
}

state_t energy_gpu_data_copy(pcontext_t *c, gpu_energy_t *data_dst, gpu_energy_t *data_src)
{
	preturn(ops.data_copy, c, data_dst, data_src);
}
