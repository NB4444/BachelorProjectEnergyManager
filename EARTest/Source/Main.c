#include <sys/types.h>
#include <ear.h>

int main() {
	ear_connect();
	
	//unsigned long a;
	//unsigned long b;
	//ear_energy(&a, &b);
	
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(0, &mask);
	ear_set_cpufreq(&mask, 10000);
	
	ear_disconnect();
	
	return 0;
}