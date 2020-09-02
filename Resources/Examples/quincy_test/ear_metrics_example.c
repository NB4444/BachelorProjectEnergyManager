/*    This program is part of the Energy Aware Runtime (EAR).
    It has been developed in the context of the BSC-Lenovo Collaboration project.
    
    Copyright (C) 2017  
    BSC Contact Julita Corbalan (julita.corbalan@bsc.es) 
        Lenovo Contact Luigi Brochard (lbrochard@lenovo.com)
*/
#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <omp.h>



#include <common/types/generic.h>
#include <metrics/cpi/cpi.h>
#include <metrics/flops/flops.h>
#include <metrics/energy/cpu.h>
#include <metrics/energy/energy_node.h>
#include <metrics/frequency/cpu.h>
#include <metrics/temperature/temperature.h>
#include <metrics/bandwidth/bandwidth.h>
#include <common/hardware/frequency.h>
#include <common/hardware/topology.h>
#include <common/system/time.h>
#include <common/system/monitor.h>





#define TEST_CPI 				1
#define TEST_FLOPS 			1
#define TEST_DC_POWER 	1
#define TEST_RAPL_POWER 1
#define TEST_RAPL_TEMP	1
#define TEST_AVG_FREQ		1
#define TEST_CPU_FREQ		1
#define TEST_RAPL_TEMP 	1
#define TEST_MEM_BW 		1

#define TOTAL_TESTS 1
#define STEPS	1



uint test = 0;
static ulong n_iterations,d_iterations;

void usage()
{
    printf("Usage: ear_metrics_example n_threads n_iterations def_frequency number_of_runs_per_test energy_plugin_path\n");
    printf("- n_threads: threads to create and bind\n");
    printf("- n_iterations: number of n_iterations to gather energy metrics\n");
    printf("- def_frequency \n");
		printf("- number_of_runs_per_test \n");
		printf("path to energy plugin\n");
		printf("This test needs root privileges, you are not allowed to run it\n");
		printf("Use export USE_MONITOR=1 to start the additional monitor thread required for dcmi energy plugin\n");
    exit(1);
}

/********************************************
 * 				Per-test auxiliary functions
 ********************************************/

int do_test_in_parallel(int t)
{
	switch (t){
	case 0: return 1;
	}
	return 1;
}

void test1()
{
	sleep(1);
}

void run_test(int index,ulong n_iterations,int cpu)
{
	int i;
	for (i=0;i<n_iterations;i++){
		switch(index){
		case 0: test1();break;
		}
	}	
	return;
}

int test_available(int i,int cpu)
{
	return 1;
}

uint get_test_ops(int index,int cpu)
{
	switch(index){
	case 0:return 1;
	}
	return 1;
}

char *get_test_name(int index,int cpu)
{
	switch(index){
  case 0:return "TEST";
  }
  return "TEST";

}

/************************************
 * 				MAIN
 ************************************/

int main (int argc, char *argv[])
{
    ull *metrics_init,*metrics_end,*metrics_diff;
		ull *mem_bd_init,*mem_bd_end,*mem_bd_diff;
    long long total_flops[8],papi_flops;
    char nodename[256];
    char buffer[1024],unc[1024];
		int steps,run,runs;	
		char *use_monitor;
		uint eunit;

    ulong start_time,end_time, exec_time;
    ulong num_ops, frequency, aux;
    long long flops, total_fp_inst;
    long long cycles, inst;
    unsigned long F,F_BASE;
		ulong secs;
		int i;

		unsigned long  dc_energy_j;
		double power_dc_w;
    double power_w, power_raw, power_raw_w,power_dram_w;
    double energy_nj, energy_j, energy_raw, energy_raw_j;
    double energy_dram_raw, energy_dram_j;
    double time_s, flops_m, flops_x_watt;
		topology_t my_topo;
		int * rapl_msr, *rapl_msr_temp_fds;
		unsigned long long *rapl_temp;
		ehandler_t my_eh;
		edata_t my_energy_data_init,my_energy_data_end;
		char *energy_plugin_path;
		size_t energy_size;
		freq_cpu_t f_init,f_end,f_diff;
		timestamp tstart,tend;
		double GBS;

    uint n_tests, n_sockets, n_threads,test_th;
    uint i_test, i_socket;
    int full_compatible,n_uncore;
    int cpu;

    if (argc != 6) {
        usage();
    }

	if (getuid()!=0) usage();

    // Options
    use_monitor=getenv("USE_MONITOR");
    n_threads = atoi(argv[1]);
    d_iterations = strtoul(argv[2], NULL, 10);
		F= (unsigned long) atoi(argv[3]);
    runs= atoi(argv[4]);
		energy_plugin_path=argv[5];

		printf("%s executing: %d threads , %d iterations at freq %lu. %d runs per test\n",
		argv[0],n_threads,d_iterations,F,runs);

		if (use_monitor != NULL){ 
			printf("Monitor initialized\n");
			if (monitor_init() != EAR_SUCCESS){
				fprintf(stderr,"Error initializing  monitor\n");
			}
		}
    // Node info
    if (topology_init(&my_topo) !=EAR_SUCCESS){
			fprintf(stderr,"Error in getting topology\n");
			exit(1);	
		}
    cpu = my_topo.model;
    gethostname(nodename, sizeof(nodename));
    full_compatible = (cpu == MODEL_SKYLAKE_X) ;
    printf("Family %d and CPU model detected %d compatible %d\n", my_topo.family,cpu,full_compatible);
    n_tests = TOTAL_TESTS;

		n_sockets = my_topo.socket_count;
		printf("Using %d cores\n",n_threads);
		#if TEST_RAPL_POWER
		fprintf(stdout,"Initializing RAPL with %d packages\n",my_topo.socket_count);
		rapl_msr=calloc(my_topo.socket_count,sizeof(int));
		metrics_init=calloc(my_topo.socket_count*RAPL_POWER_EVS,sizeof(unsigned long long));
		metrics_end=calloc(my_topo.socket_count*RAPL_POWER_EVS,sizeof(unsigned long long));
		metrics_diff=calloc(my_topo.socket_count*RAPL_POWER_EVS,sizeof(unsigned long long));
    if (init_rapl_msr(rapl_msr)<0){
			fprintf(stderr,"Error in RAPL initializtion\n");	
			exit(1);
		}
		#endif
		rapl_temp=calloc(my_topo.socket_count,sizeof(unsigned long long));
		#if TEST_RAPL_TEMP
		fprintf(stdout,"Initializing Temperature \n");
		rapl_msr_temp_fds=calloc(my_topo.socket_count,sizeof(int));
		init_temp_msr(rapl_msr_temp_fds);
		#endif
		#if TEST_CPI
		fprintf(stdout,"Initializing CPI\n");
    if (init_basic_metrics() != EAR_SUCCESS){
			fprintf(stderr,"Error in CPI metric initialization\n");
			exit(1);
		}
		#endif
		#if TEST_FLOPS
		fprintf(stdout,"Initializing FLOPS\n");
    if (full_compatible)
    {
			fprintf(stdout,"System full compatible\n");
    	if (init_flops_metrics() == 0){
      	fprintf(stderr,"Error in FLOPS metric initialization\n");
      	exit(1);
    	}
    }
		#endif
		n_uncore = 1;
		#if TEST_MEM_BW
		fprintf(stdout,"Initializing Mem. BW\n");
		n_uncore=init_uncores(my_topo.model);
		reset_uncores();
		#endif		
		alloc_uncores(&mem_bd_init,n_uncore);
		alloc_uncores(&mem_bd_end,n_uncore);
		alloc_uncores(&mem_bd_diff,n_uncore);
		#if TEST_CPU_FREQ
		fprintf(stdout,"Initializing CPU frequency\n");
		frequency_init(my_topo.cpu_count);
		F_BASE=frequency_get_cpu_freq(0);
    printf("Default frequency was %lu\n",F_BASE);
		printf("Setting frequency to %lu\n",F);
    frequency_set_all_cpus(F);
		#endif
		#if TEST_AVG_FREQ	
		fprintf(stdout,"Initializing AVG CPU computation\n");
		if (freq_cpu_init(&my_topo) != EAR_SUCCESS){
			fprintf(stderr,"Error Initializing AVG CPU computation\n");
			exit(1);
		}
		freq_cpu_data_alloc(&f_init,NULL,NULL);
		freq_cpu_data_alloc(&f_end,NULL,NULL);
		freq_cpu_data_alloc(&f_diff,NULL,NULL);
		#endif
		#if TEST_DC_POWER	
		fprintf(stdout,"Initializing Node energy readings\n");

		if (energy_load(energy_plugin_path) != EAR_SUCCESS){
			fprintf(stderr,"Error loading energy plugin");
			exit(1);
		}
		if (energy_initialization(&my_eh) != EAR_SUCCESS){
      fprintf(stderr,"Error initializing energy node");
      exit(1);
    }
		if (energy_datasize(&my_eh,&energy_size)!= EAR_SUCCESS){
      fprintf(stderr,"Error getting energy datasize");
      exit(1);
    }
		fprintf(stderr,"Energy datasiz is %d\n",energy_size);
		my_energy_data_init=malloc(energy_size);
		my_energy_data_end=malloc(energy_size);
		fprintf(stdout,"Memory for energy allocated\n");
		energy_units(&my_eh,&eunit);
		#endif
		n_iterations=d_iterations*10;
		// We ran all the different number of iterations
	for (steps=0;steps<STEPS;steps++)
	{
	n_iterations=(unsigned long)(n_iterations/10);
	printf("Testing with %lu iterations\n",n_iterations);	
	printf("-----------------------------\n");
    // Creating the threads
    for (i_test = 0; i_test < TOTAL_TESTS; ++i_test)
    {
        printf("Starting test %d\n",i_test);
		for (run=0;run<runs;run++)
		{
			printf("---------- RUN %d for test %d -----------\n",run,i_test);
      test = i_test;
      energy_raw = 0;
      energy_dram_raw = 0;
			dc_energy_j=0;
			if (test_available(i_test,cpu)){
        #pragma omp parallel firstprivate(run,test,i_test,n_iterations,cpu)  num_threads(n_threads) if (do_test_in_parallel(i_test))
        {
				#pragma omp barrier
        if ((omp_get_thread_num() == 0) || !do_test_in_parallel(test))
        {
				printf("Starting metrics\n");	
        energy_raw=0;
        energy_dram_raw=0;
        secs=0;
        #if TEST_RAPL_POWER
        read_rapl_msr(rapl_msr,metrics_init);
        #endif
        if (full_compatible)
        {
          #if TEST_FLOPS
          start_flops_metrics();
          #endif
        }
        #if TEST_AVG_FREQ
        freq_cpu_read(&f_init);
        #endif
        #if TEST_CPI
        start_basic_metrics();
        #endif
        #if TEST_MEM_BW
        stop_uncores(mem_bd_init);
        start_uncores();
        #endif
        #if TEST_DC_POWER
        if (energy_dc_time_read(&my_eh,my_energy_data_init,&start_time) != EAR_SUCCESS){
          fprintf(stderr,"Error reading energy\n");
        }
        #else
        timestamp_get(&tstart);
        #endif
        memset(rapl_temp,0,sizeof(unsigned long long)*n_sockets);
        #if TEST_RAPL_TEMP
        read_temp_msr(rapl_msr_temp_fds,rapl_temp);
        #endif

        }
			
			// Run the test

        run_test(test,n_iterations,cpu);

        if ((omp_get_thread_num() == 0) || !do_test_in_parallel(test))
        {
				// Get metrics
				printf("Stopping metrics\n");
				#if TEST_DC_POWER
        if (energy_dc_time_read(&my_eh,my_energy_data_end,&end_time) != EAR_SUCCESS){
          fprintf(stderr,"Error reading energy\n");
        }
        exec_time = (end_time - start_time);
        #else
        timestamp_get(&tend);
        exec_time = timestamp_diff(&tend,&tstart,TIME_MSECS);
        #endif
        cycles=0;inst=0;
        #if TEST_CPI
        stop_basic_metrics(&cycles,&inst);
        reset_basic_metrics();
        #endif
        #if TEST_MEM_BW
        stop_uncores(mem_bd_end);
        #endif
        #if TEST_AVG_FREQ
        freq_cpu_read(&f_end);
        #endif
        /* Reading and computing RAPL energy values */
        #if TEST_RAPL_POWER
        read_rapl_msr(rapl_msr,metrics_end);
        #endif
        if (full_compatible)
        {
          memset(total_flops,0,sizeof(total_flops));
          papi_flops=0;
          #if TEST_FLOPS
          stop_flops_metrics(&papi_flops,&total_flops[0]);
          reset_flops_metrics();
          #endif
        }

        }
        } // PARALLEL
				#if TEST_DC_POWER
				/* Node energy */
				energy_accumulated(&my_eh,&dc_energy_j,my_energy_data_init,my_energy_data_end);
				dc_energy_j = dc_energy_j / eunit;
				#else
				dc_energy_j = 0;
				#endif
        // Exec time in seconds
        time_s = ((double) exec_time) / (double) 1000;
				/* FP operations */
				total_fp_inst=0;
        if (full_compatible)
        {
						#if TEST_FLOPS
            total_fp_inst = total_flops[0] + total_flops[1] + total_flops[2] + total_flops[3] +
                    total_flops[4] + total_flops[5] + total_flops[6] + total_flops[7];
						#endif
        }
				energy_raw=0;	
				energy_dram_raw=0;
				power_raw_w = 0;
				power_dram_w = 0;
				memset(metrics_diff,0,sizeof(unsigned long long)*n_sockets);
				#if TEST_RAPL_POWER
				/* RAPL energy */
				diff_rapl_msr_energy(metrics_diff,metrics_end,metrics_init);
        for (i_socket = 0; i_socket < n_sockets; ++i_socket)
        {
            // Energy per socket in nano juls
						/* DRAM 0, DRAM 1,..DRAM N, PCK0,PCK1,...PCKN  */
            energy_raw += (double) metrics_diff[n_sockets*RAPL_PCK_EV + i_socket];
            energy_dram_raw += (double) metrics_diff[n_sockets*RAPL_DRAM_EV+i_socket];
        }
        // Raw data
        energy_raw_j = energy_raw / (double) 1000000000;
        power_raw_w = energy_raw_j / time_s;

        energy_dram_j = energy_dram_raw / (double) 1000000000;
        power_dram_w = energy_dram_j / time_s;
				#endif
				frequency=0;
				#if TEST_AVG_FREQ
				freq_cpu_data_diff(&f_end,&f_init,NULL,&frequency);
				#endif


				power_dc_w = 0;
				#if TEST_DC_POWER
				power_dc_w=(double) dc_energy_j / (double) time_s;
				#endif
				GBS = 0.0;
				sprintf(unc,"no_unc_test");
				#if TEST_MEM_BW
				diff_uncores(mem_bd_diff,mem_bd_end,mem_bd_init,n_uncore);
				compute_mem_bw(mem_bd_end,mem_bd_init,&GBS,time_s,n_uncore);
				uncores_to_str(mem_bd_diff,n_uncore,unc,sizeof(unc));
				#endif

        num_ops = (ull) get_test_ops(test, cpu) * n_iterations * n_threads;
        // Floating point operations per second
        flops_m = ((double) num_ops) / time_s;
        // Flops to Mflops
        flops_m = flops_m / (double) 1000000000;
        // Mflops x watt
        flops_x_watt = 0;
				#if TEST_DC_POWER
        flops_x_watt = flops_m / power_dc_w;
				#endif

        printf("TEST (%d): %s\n", i_test, get_test_name(test,cpu));
				printf("\tCPI %.3lf\n",(double) cycles / (double) inst);
        printf("\tEnergy RAPL pck(j): %.2lf\n", energy_raw_j);
        printf("\tEnergy RAPL dram(j): %.2lf\n", energy_dram_j);
        printf("\tEnergy node(j): %lu\n", dc_energy_j);
        printf("\tAvg. Power RAPL pckg (W): %.2lf\n", power_raw_w);
        printf("\tAvg. Power RAPL dram (W) %.2lf\n", power_dram_w);
        printf("\tAvg. Power dc (W): %.2lf\n", power_dc_w);
        printf("\tTime (s): %.2lf\n", time_s);
        printf("\t(Gflops_code x n_thread)/dc_power_watt: %.3lf\n", flops_x_watt);
        printf("\t(Gflops_met x n_thread)/dc_power_watt: %.3lf\n", (float)(papi_flops*n_threads)/(float)(power_dc_w*1000000000));
        printf("\tTotal FP_OPS x n_threads(millions): based on test %lu, per_th %lu\n", num_ops/1000000,num_ops/(n_threads*1000000));
        printf("\tTotal FOP: flops_metrics(millions)=%lld - fp_inst=%lld\n", papi_flops/1000000,total_fp_inst/1000000);
        printf("\tAvg. Frequency : %.2f GHz\n", (float)frequency/1000000.0);
				printf("\tMem. Bandwith %.2lf GB/s\n",GBS);
				for (i=0;i<my_topo.socket_count;i++){
					printf("\tSocket [%d] Power PCK %.2f DRAM %.2f Temp %llu\n",i,(float)metrics_diff[n_sockets*RAPL_PCK_EV + i]/(float)(time_s*1000000000),
					(float)metrics_diff[n_sockets*RAPL_DRAM_EV + i]/(float)(time_s*1000000000),rapl_temp[i]);
				}
				printf("\tUncore counters %s \n",unc);
				printf("\tFlops[0]=%lld Flops[1]=%lld Flops[2]=%lld Flops[3]=%lld Flops[4]=%lld Flops[5]=%lld Flops[6]=%lld Flops[7]=%lld\n",
				total_flops[0],total_flops[1],total_flops[2],total_flops[3],total_flops[4],total_flops[5],total_flops[6],total_flops[7]);	


        if (full_compatible)
        {
            printf("\tAVX512 percentage: %lf\n", (double) total_flops[7] / (double) inst);
            printf("\tFP percentage: %lf\n", (double) total_fp_inst / (double) inst);
            printf("\ttotal instructions(millions): %llu\n", inst/1000000);
            printf("\ttotal cycles(millions): %llu\n",  cycles/1000000);
        }

		}
    } // for (run
	} // for (test
	} // for (step
	#if TEST_CPU_FREQ
  frequency_set_all_cpus(F_BASE);
  frequency_dispose();
	#endif
	#if TEST_DC_POWER
  energy_dispose(&my_eh);
	#endif
	#if TEST_MEM_BW
	dispose_uncores();
	#endif

  return 0;
}
