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

#ifndef COMMON_APIS_SUSCRIPTOR_H
#define COMMON_APIS_SUSCRIPTOR_H

#include "apis.h"
#include "common/types.h"
#include "common/states.h"

#define sus(context) \
		((suscription_t *) context->suscription)

#define suscribed(context) \
		context->suscribed == 1

typedef state_t (*read_f) (api_ctx_t *, void *);
typedef state_t (*suscription_f) (void *);

typedef struct suscription_s {
	suscription_f	 suscribe;
	api_ctx_t		*api_cntx;
    read_f			 api_call;
    void			*api_memm;
    uint 	 	 	 api_time;
    uint		 	 id;
} suscription_t;

state_t suscriptor_init();

state_t suscriptor_suscribe(void *suscription);

state_t suscriptor_burst(void *suscription, ullong time_ms);

state_t suscriptor_relax(void *suscription);

suscription_t *suscription();

struct monitor_s {
	suscription_t *(*suscription) ();
}
Monitor __attribute__((weak)) = {
	.suscription = suscription
};

#endif //EAR_STASH_MONITOR_H