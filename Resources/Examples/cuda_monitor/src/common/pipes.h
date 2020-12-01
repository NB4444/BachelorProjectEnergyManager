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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "states.h"
#include "time.h"

typedef struct pipe_s {
	fd_set fds_select;
	int connected;
	int state;
	int fd;
} pipe_t;

#define PIPES_RO O_RDONLY
#define PIPES_WO O_WRONLY
#define PIPES_RW O_RDWR

#define pipes_ready(p) p.state == 1

state_t pipes_open(pipe_t* p, const char* pathname, int type, int open);

state_t pipes_close(pipe_t* p);

state_t pipes_read(pipe_t* p, void* buffer, size_t size);

state_t pipes_write(pipe_t* p, void* buffer, size_t size);

state_t pipes_select(pipe_t* p, ulong timeout_s);
