SRCDIR = $(HOME)/gitlab/ear_private/src


CC = icc
MACHINE_CFLAGS = -qopenmp 
CFLAGS = -I$(SRCDIR) -I. $(MACHINE_CFLAGS) -rdynamic 



######## FILES


METRICS = $(SRCDIR)/metrics/libmetrics.a
COMMON =  $(SRCDIR)/common/libcommon.a


BINARIES = ear_metrics_example

######## RULES

all: $(BINARIES) 

ear_metrics_example:ear_metrics_example.c $(METRICS) $(COMMON) 
	$(CC) $(CFLAGS) -o ear_metrics_example ear_metrics_example.c  $(METRICS) $(COMMON) -pthread


	

######## OPTIONS

clean:
	rm -f $(BINARIES)
	rm -f *.o *.a *.so
