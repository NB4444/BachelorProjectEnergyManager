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

#include <error.h>
#include <string.h>
#include <pthread.h>
#include "common/time.h"
#include "common/types.h"
#include "common/states.h"
#include "apis_suscriptor.h"

typedef struct suscribers_s {
	suscription_t	suscription;
	timestamp_t		time_since;
	ullong 		 	time_limit;
	ullong		 	time_passed;
	ullong		 	time_follow;
	int			 	ready;
} suscribers_t;

static pthread_t	thread;
static suscribers_t	data[32];
static uint			enabled;
static uint			d;

#define debug(...)

static void monitor_sleep()
{
	timestamp_t next_wake_ts;
	ullong next_wake_ms;
	suscribers_t *s;
	state_t r;
	int i;

	if (d > 0) {
		next_wake_ms = ULLONG_MAX;
	} else {
		// If no suscriptions, wake up every second
		next_wake_ms = 1000ULL;
	}

	for (i = 0; i < d; ++i)
	{
		s = &data[i];
		//
		if (s->time_follow < next_wake_ms) {
			next_wake_ms = s->time_follow;
		}
	}

	timestamp_revert(&next_wake_ts, &next_wake_ms, TIME_MSECS);
	debug("going to sleep %llu ms (%ld, %ld)",
		next_wake_ms, next_wake_ts.tv_sec, next_wake_ts.tv_nsec);
	
	if ((r = nanosleep(&next_wake_ts, NULL)) != 0) {
		//
	}
}

static void monitor_time_calc(suscribers_t *s)
{
	timestamp_t time_aux;
	//
	time_aux = s->time_since;
	//
	timestamp_getfast(&s->time_since);
	
	debug("-- time since %llu, time aux %llu ms",
		timestamp_convert(&s->time_since, TIME_MSECS),
		timestamp_convert(&time_aux, TIME_MSECS));
	//
	s->time_passed += timestamp_diff(&s->time_since, &time_aux, TIME_MSECS);
	//
	s->time_follow  = s->time_limit - s->time_passed;
	//
	debug("-- time limit %llu, time passed %llu ms, time next %lld",
		s->time_limit, s->time_passed, s->time_follow);
	//
	if (s->time_passed >= s->time_limit) {
		s->time_follow = s->time_limit;
		s->time_passed = 0;
		s->ready = 1;
	}
	
	debug("-- ready %u", s->ready);
}

static void *monitor(void *p)
{
	suscription_t *sp;
	suscribers_t *sb;
	int i;

	while (enabled)
	{
		for (i = 0; i < d; ++i)
		{
			sb = &data[i];
			sp = &sb->suscription;
			//
			debug("calculing suscription %d:", i);
			monitor_time_calc(sb);

			if (sb->ready) {
				sp->api_call(sp->api_cntx, sp->api_memm);
				sb->ready = 0;
			}
		}

		monitor_sleep();
	}
	return EAR_SUCCESS;
}

state_t suscriptor_init()
{
	int errno;

	if (enabled != 0) {
		return_msg(EAR_ERROR, "monitor already enabled");
	}

	//
	errno = pthread_create(&thread, NULL, monitor, NULL);
	enabled = (errno == 0);

	if (!enabled) {
		return_msg(EAR_ERROR, strerror(errno));
	}

	return EAR_SUCCESS;
}

state_t suscriptor_suscribe(void *_s)
{
	//suscription_t *s = &data[d].suscription;
	suscription_t *n = (suscription_t *) _s;

	if (_s          == NULL) return_msg(EAR_BAD_ARGUMENT, "the suscription can't be NULL");
	if (n->api_time == 0   ) return_msg(EAR_BAD_ARGUMENT, "time cant be zero");
	if (n->api_cntx == NULL) return_msg(EAR_BAD_ARGUMENT, "context is NULL");
	if (n->api_call == NULL) return_msg(EAR_BAD_ARGUMENT, "reading call is NULL"); 
	if (n->api_memm == NULL) return_msg(EAR_BAD_ARGUMENT, "reading memory is NULL"); 

	// Time processing
	timestamp_getfast(&data[d].time_since);
	data[d].time_limit  = n->api_time;
	data[d].time_passed = data[d].time_limit;
	data[d].time_follow = 0;
	// Description
	debug("suscribed for %llu ms", data[d].time_limit);
	// Its ok, go ahead
	n->api_cntx->suscription = n;
	n->api_cntx->suscribed   = 1;
	n->id = d;
	d    += 1;

	return EAR_SUCCESS;
}

state_t suscriptor_burst(void *_s, ullong time_ms)
{
	suscription_t *s = (suscription_t *) _s;

	if (s == NULL) {
		return_msg(EAR_BAD_ARGUMENT, "the suscription can't be NULL");
	}
	if (time_ms == 0) {
		return_msg(EAR_BAD_ARGUMENT, "time cant be zero");
	}
	
	data[s->id].time_limit = time_ms;
	return EAR_SUCCESS;
}

state_t suscriptor_relax(void *_s)
{
	suscription_t *s = (suscription_t *) _s;

	if (s == NULL) {
		return_msg(EAR_BAD_ARGUMENT, "the suscription can't be NULL");
	}
	
	data[s->id].time_limit = data[s->id].suscription.api_time;
	return EAR_SUCCESS;
}

suscription_t *suscription()
{
	data[d].suscription.suscribe = suscriptor_suscribe;
	return &data[d].suscription;
}
