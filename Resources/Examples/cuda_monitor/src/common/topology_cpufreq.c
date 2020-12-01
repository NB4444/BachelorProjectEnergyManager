#include <common/sizes.h>
#include <common/states.h>
#include <common/types.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static ulong read_freq(int fd) {
	char freqc[8];
	ulong freq;
	int i = 0;
	char c;

	while((read(fd, &c, sizeof(char)) > 0) && ((c >= '0') && (c <= '9'))) {
		freqc[i] = c;
		i++;
	}

	freqc[i] = '\0';
	freq = atoi(freqc);
	return freq;
}

state_t topology_cpufreq_getbase(uint cpu, ulong* freq_base) {
	char path[1024];
	int fd;

	sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_available_frequencies", cpu);

	if((fd = open(path, O_RDONLY)) < 0) {
		return EAR_ERROR;
	}

	ulong f0 = read_freq(fd);
	ulong f1 = read_freq(fd);
	ulong fx = f0 - f1;

	if(fx == 1000) {
		*freq_base = f1;
	} else {
		*freq_base = f0;
	}

	close(fd);

	return EAR_SUCCESS;
}
