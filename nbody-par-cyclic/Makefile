CFLAGS	+= -Wall
CFLAGS	+= -O3
CFLAGS	+= -g2
MPICC = mpic++
MPIRUN = mpirun

nbody-par: nbody-par.c
	$(MPICC) $(CFLAGS) -o nbody-par nbody-par.c -lm

clean:
	rm -f *.o nbody-par *~ *core
