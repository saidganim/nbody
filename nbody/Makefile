CFLAGS	+= -Wall
CFLAGS	+= -O3
CFLAGS	+= -g2

nbody-seq: nbody-seq.c
	gcc $(CFLAGS) -o nbody-seq nbody-seq.c -lm

clean:
	rm -f *.o nbody-seq *~ *core
