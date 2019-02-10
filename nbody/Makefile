CFLAGS	+= -Wall
CFLAGS	+= -O3
CFLAGS	+= -g2
CP := cp

all: nbody-seq
	$(MAKE) -C nbody-par_cyclic_omp/
	$(CP) nbody-par_cyclic_omp/nbody-par ./


nbody-seq: nbody-seq.c
	gcc $(CFLAGS) -o nbody-seq nbody-seq.c -lm


clean:
	rm -f *.o nbody-seq *~ *core
