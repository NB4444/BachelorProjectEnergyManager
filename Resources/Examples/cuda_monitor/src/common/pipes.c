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

#include "pipes.h"

static char buffer[4096];

state_t pipes_open(pipe_t *p, const char *pathname, int type, int _open) {
  if (_open) {
    sprintf(buffer, "rm %s", pathname);
    system(buffer);

    if (mkfifo(pathname, 0700) < 0) {
      return_msg(EAR_ERROR, strerror(errno));
    }
  }

  p->fd = open(pathname, type | O_NONBLOCK);

  if (p->fd < 0) {
    return_msg(EAR_ERROR, strerror(errno));
  }

  chmod(pathname, S_IRUSR | S_IWUSR | S_IWGRP | S_IRGRP | S_IROTH | S_IWOTH);
  p->connected = 1;

  return EAR_SUCCESS;
}

state_t pipes_close(pipe_t *p) {
  close(p->fd);
  return EAR_SUCCESS;
}

state_t pipes_read(pipe_t *p, void *buffer, size_t size) {
  size_t s;

  if ((s = read(p->fd, buffer, size)) != size) {
    return_msg(EAR_ERROR, strerror(errno));
  }
  return EAR_SUCCESS;
}

state_t pipes_write(pipe_t *p, void *buffer, size_t size) {
  size_t s;
  if ((s = write(p->fd, buffer, size)) != size) {
    return_msg(EAR_ERROR, strerror(errno));
  }
  return EAR_SUCCESS;
}

state_t pipes_select(pipe_t *p, ulong timeout_ms) {
  struct timeval timeout;

  //
  timeout.tv_sec = 0L;
  timeout.tv_usec = timeout_ms * 1000L;

  //
  FD_ZERO(&p->fds_select);
  FD_SET(p->fd, &p->fds_select);

  if ((p->state = select(p->fd + 1, &p->fds_select, NULL, NULL, &timeout)) ==
      -1) {
    return_msg(EAR_ERROR, strerror(errno));
  }

  return EAR_SUCCESS;
}
