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

struct tp_select_s
{
	int core;
	int l3;
	int socket;
}
	TPSelect __attribute__((weak)) =
{
	.core = 1,
	.l3 = 2,
	.socket = 3,
};

struct tp_group_s
{
	int value;
	int merge;
}
	TPGroup __attribute__((weak)) =
{
	.value = 1,
	.merge = 2,
};

typedef struct core_s
{
	int id;
	int is_thread;
	int l3_id;
	int sibling_id;
	int socket_id;
	ulong freq_base;
} core_t;

typedef struct topology_s
{
	core_t *cpus;
	int cpu_count;
	int thread_count;
	int core_count;
	int socket_count;
	int threads_per_core;
	int smt_enabled;
	int l3_count;
} topology_t;

state_t topology_select(topology_t *t, topology_t *s, int component, int group, int val);

state_t topology_init(topology_t *topo);

state_t topology_close(topology_t *topo);
