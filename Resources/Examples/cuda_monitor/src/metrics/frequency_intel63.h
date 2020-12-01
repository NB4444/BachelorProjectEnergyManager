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

#ifndef FREQUENCY_INTEL63_H
#define FREQUENCY_INTEL63_H

#include <common/msr.h>
#include <common/states.h>
#include <common/topology.h>
#include <common/types.h>

#define MSR_IA32_APERF 0x000000E8
#define MSR_IA32_MPERF 0x000000E7

typedef struct frequency_effective_s {
	void* data;
} frequency_effective_t;

state_t freq_intel63_init(frequency_effective_t* ef, topology_t* tp);

state_t freq_intel63_dispose(frequency_effective_t* ef);

state_t freq_intel63_read(frequency_effective_t* ef, ulong* freq);

state_t freq_intel63_read_count(frequency_effective_t* ef, uint* count);

#endif // FREQUENCY_INTEL63_H
