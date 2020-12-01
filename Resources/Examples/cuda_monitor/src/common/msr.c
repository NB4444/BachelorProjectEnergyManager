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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "msr.h"
#include "sizes.h"
#include "states.h"

#define MSR_MAX 4096
static int counters[MSR_MAX];
static int fds[MSR_MAX];

/* */
state_t msr_open(uint cpu) {
	char msr_file_name[SZ_PATH_KERNEL];

	if(cpu >= MSR_MAX) {
		return EAR_BAD_ARGUMENT;
	}

	if(counters[cpu] == 0) {
		sprintf(msr_file_name, "/dev/cpu/%d/msr", cpu);
		fds[cpu] = open(msr_file_name, O_RDWR);
	}

	if(fds[cpu] < 0) {
		return EAR_OPEN_ERROR;
	}

	counters[cpu] += 1;

	return EAR_SUCCESS;
}

/* */
state_t msr_close(uint cpu) {
	if(cpu >= MSR_MAX) {
		return EAR_BAD_ARGUMENT;
	}

	if(counters[cpu] == 0) {
		return EAR_ALREADY_CLOSED;
	}

	counters[cpu] -= 1;

	if(counters[cpu] == 0) {
		close(fds[cpu]);
	}

	return EAR_SUCCESS;
}

/* */
state_t msr_read(uint cpu, void* buffer, size_t size, off_t offset) {
	if(cpu >= MSR_MAX) {
		return EAR_BAD_ARGUMENT;
	}

	if(counters[cpu] == 0) {
		return EAR_NOT_INITIALIZED;
	}

	if(pread(fds[cpu], buffer, size, offset) != size) {
		return EAR_READ_ERROR;
	}

	return EAR_SUCCESS;
}

/* */
state_t msr_write(uint cpu, const void* buffer, size_t size, off_t offset) {
	if(cpu >= MSR_MAX) {
		return EAR_BAD_ARGUMENT;
	}

	if(counters[cpu] == 0) {
		return EAR_NOT_INITIALIZED;
	}

	if(pwrite(fds[cpu], buffer, size, offset) != size) {
		return EAR_ERROR;
	}

	return EAR_SUCCESS;
}
