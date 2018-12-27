/*  
    N-Body simulation code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <mpi.h>
#include <omp.h>

#define GRAVITY     1.1
#define FRICTION    0.01
#define MAXBODIES   10000
#define DELTA_T     (0.025/5000)
#define BOUNCE      -0.9
#define SEED        27102015

int world_rank;
int P;
MPI_Status status;
MPI_Request request;
int chunk_size;
int lborder, rborder;
double *forces;
double *forces2;
double *coords;
double *coords2;

struct bodyType {
    double x[2];        /* Old and new X-axis coordinates */
    double y[2];        /* Old and new Y-axis coordinates */
    double xf;          /* force along X-axis */
    double yf;          /* force along Y-axis */
    double xv;          /* velocity along X-axis */
    double yv;          /* velocity along Y-axis */
    double mass;        /* Mass of the body */
    double radius;      /* width (derived from mass) */
};


struct world {
    struct bodyType bodies[MAXBODIES];
    int                 bodyCt;
    int                 old;    // Flips between 0 and 1

    /*  Dimensions of space (very finite, ain't it?) */
    int                 xdim;
    int                 ydim;
};

/*  Macros to hide memory layout
*/
#define X(w, B)        (w)->bodies[B].x[(w)->old]
#define XN(w, B)       (w)->bodies[B].x[(w)->old^1]
#define Y(w, B)        (w)->bodies[B].y[(w)->old]
#define YN(w, B)       (w)->bodies[B].y[(w)->old^1]
#define XF(w, B)       (w)->bodies[B].xf
#define YF(w, B)       (w)->bodies[B].yf
#define XV(w, B)       (w)->bodies[B].xv
#define YV(w, B)       (w)->bodies[B].yv
#define R(w, B)        (w)->bodies[B].radius
#define M(w, B)        (w)->bodies[B].mass

#define MIN(a,b) ({__typeof__(a) _a = a; __typeof__(b) _b = b;\
                    _a < _b? _a : _b;})

static inline int on_master(){
  return world_rank == 0;
}

static void clear_forces(struct world *world){
    /* Clear force accumulation variables */
    #pragma omp parallel for
    for (int b = lborder; b < world->bodyCt; ++b) {
        YF(world, b) = XF(world, b) = 0;
    }
}

void gather_coords(struct world* world){
    #pragma omp parallel for
    for(int i = lborder; i < rborder; ++i){
        coords[(i - lborder) * 2] = X(world, i);
        coords[(i - lborder) * 2 + 1] = Y(world, i);
    }
    // Sending own coordinates to all revious process (VERY INEFFICIENT)
    for(int i = 0; i < world_rank; ++i){
        MPI_Isend(coords, chunk_size * 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request);
    }

    // Receiving coordinates of other process
    for(int i = world_rank + 1; i < P; ++i){
        MPI_Recv(coords2, chunk_size * 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
        for(int j = i * chunk_size; j < (i + 1) * chunk_size; ++j){
            X(world, j) = coords2[(j - i * chunk_size) * 2];
            Y(world, j) = coords2[(j - i * chunk_size) * 2 + 1];
        }
    }
}

void gather_coords_bcast(struct world* world){ // gathering coordinates via MPI_Bcast( might save messages in the network)
    // Sending own coordinates to all revious process
    for(int i = 0; i < P; ++i){
        if(i == world_rank)
             for(int j = lborder; j < rborder; ++j){
                coords[(j - lborder) * 2] = X(world, j);
                coords[(j - lborder) * 2 + 1] = Y(world, j);
            }
        MPI_Bcast(coords, chunk_size * 2, MPI_DOUBLE, i, MPI_COMM_WORLD);
        if(i != world_rank){
            for(int j = i * chunk_size; j < (i + 1) * chunk_size; ++j){
                X(world, j) = coords[(j - i * chunk_size) * 2];
                Y(world, j) = coords[(j - i * chunk_size) * 2 + 1];
            }
        }
    }
}

void gather_forces(struct world* world){
    #pragma omp parallel for // cache false sharing is possible here
    for(int i = lborder; i < world->bodyCt; ++i){
        forces[i * 2] = XF(world, i);
        forces[i * 2 + 1] = YF(world, i);
    }
    // Sending own coordinates to all revious process (VERY INEFFICIENT)
    for(int i = world_rank + 1; i < P; ++i){
        MPI_Isend(forces, world->bodyCt * 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request);
    }

    // Receiving coordinates of other process
    for(int i = 0; i < world_rank; ++i){
        MPI_Recv(forces2, world->bodyCt * 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
        for(int j = lborder; j < rborder; ++j){
            XF(world, j) += forces2[j * 2];
            YF(world, j) += forces2[j * 2 + 1];
        }
    }
}


void gather_forces_reduce(struct world* world){
    memset(forces, 0x0, sizeof(double) * world->bodyCt * 2);
    #pragma omp parallel for
    for(int i = lborder; i < world->bodyCt; ++i){
        forces[i * 2] = world->bodies[i].xf;
        forces[i * 2 + 1] = world->bodies[i].yf;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // gathering forces from all the nodes to all the nodes :)
    MPI_Allreduce(forces, forces2, world->bodyCt * 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #pragma omp parallel for
    for(int i = 0; i < world->bodyCt; ++i){
        world->bodies[i].xf = forces2[i * 2];
        world->bodies[i].yf = forces2[i * 2 + 1];
    }
}

static void compute_forces(struct world *world){
    /* Incrementally accumulate forces from each body pair,
       skipping force of body on itself (c == b)
    */
    #pragma omp parallel for
    for (int b = lborder; b < rborder; ++b) {
        for (int c = b + 1; c < world->bodyCt; ++c) {
            double dx = X(world, c) - X(world, b);
            double dy = Y(world, c) - Y(world, b);
            double angle = atan2(dy, dx);
            double dsqr = dx*dx + dy*dy;
            double mindist = R(world, b) + R(world, c);
            double mindsqr = mindist*mindist;
            double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
            double force = M(world, b) * M(world, c) * GRAVITY / forced;
            double xf = force * cos(angle);
            double yf = force * sin(angle);

            /* Slightly sneaky...
               force of b on c is negative of c on b;
            */
           #pragma omp critical 
           {
            XF(world, b) += xf;
            YF(world, b) += yf;
            XF(world, c) -= xf;
            YF(world, c) -= yf;
           }
            
        }
    }
}

static void compute_velocities(struct world *world){
    #pragma omp parallel for
    for (int b = lborder; b < rborder; ++b) {
        double xv = XV(world, b);
        double yv = YV(world, b);
        double force = sqrt(xv*xv + yv*yv) * FRICTION;
        double angle = atan2(yv, xv);
        double xf = XF(world, b) - (force * cos(angle));
        double yf = YF(world, b) - (force * sin(angle));

        XV(world, b) += (xf / M(world, b)) * DELTA_T;
        YV(world, b) += (yf / M(world, b)) * DELTA_T;
    }
}

static void compute_positions(struct world *world){
    #pragma omp parallel for    
    for (int b = lborder; b < rborder; ++b) {
        double xn = X(world, b) + XV(world, b) * DELTA_T;
        double yn = Y(world, b) + YV(world, b) * DELTA_T;

        /* Bounce off image "walls" */
        if (xn < 0) {
            xn = 0;
            XV(world, b) = -XV(world, b);
        } else if (xn >= world->xdim) {
            xn = world->xdim - 1;
            XV(world, b) = -XV(world, b);
        }
        if (yn < 0) {
            yn = 0;
            YV(world, b) = -YV(world, b);
        } else if (yn >= world->ydim) {
            yn = world->ydim - 1;
            YV(world, b) = -YV(world, b);
        }

        /* Update position */
        XN(world, b) = xn;
        YN(world, b) = yn;
    }
}


/*  Graphic output stuff...
 */

#include <fcntl.h>
#include <sys/mman.h>

struct filemap {
    int            fd;
    off_t          fsize;
    void          *map;
    unsigned char *image;
};


static void filemap_close(struct filemap *filemap){
    if (filemap->fd == -1) {
        return;
    }
    close(filemap->fd);
    if (filemap->map == MAP_FAILED) {
        return;
    }
    munmap(filemap->map, filemap->fsize);
}


static unsigned char *Eat_Space(unsigned char *p){
    while ((*p == ' ') ||
           (*p == '\t') ||
           (*p == '\n') ||
           (*p == '\r') ||
           (*p == '#')) {
        if (*p == '#') {
            while (*(++p) != '\n') {
                // skip until EOL
            }
        }
        ++p;
    }

    return p;
}


static unsigned char *Get_Number(unsigned char *p, int *n){
    p = Eat_Space(p);  /* Eat white space and comments */

    int charval = *p;
    if ((charval < '0') || (charval > '9')) {
        errno = EPROTO;
        return NULL;
    }

    *n = (charval - '0');
    charval = *(++p);
    while ((charval >= '0') && (charval <= '9')) {
        *n *= 10;
        *n += (charval - '0');
        charval = *(++p);
    }

    return p;
}


static int map_P6(const char *filename, int *xdim, int *ydim, struct filemap *filemap){
    /* The following is a fast and sloppy way to
       map a color raw PPM (P6) image file
    */
    int maxval;
    unsigned char *p;

    /* First, open the file... */
    if ((filemap->fd = open(filename, O_RDWR)) < 0) {
        goto ppm_abort;
    }

    /* Read size and map the whole file... */
    filemap->fsize = lseek(filemap->fd, (off_t)0, SEEK_END);
    filemap->map = mmap(0,                      // Put it anywhere
                        filemap->fsize,         // Map the whole file
                        (PROT_READ|PROT_WRITE), // Read/write
                        MAP_SHARED,             // Not just for me
                        filemap->fd,            // The file
                        0);                     // Right from the start
    if (filemap->map == MAP_FAILED) {
        goto ppm_abort;
    }
    

    /* File should now be mapped; read magic value */
    p = (unsigned char*)filemap->map;
    if (*(p++) != 'P') goto ppm_abort;
    switch (*(p++)) {
    case '6':
        break;
    default:
        errno = EPROTO;
        goto ppm_abort;
    }

    p = Get_Number(p, xdim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, ydim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, &maxval);         // Get image max value */
    if (p == NULL) goto ppm_abort;

    /* Should be 8-bit binary after one whitespace char... */
    if (maxval > 255) {
        goto ppm_abort;
    }
    if ((*p != ' ') &&
        (*p != '\t') &&
        (*p != '\n') &&
        (*p != '\r')) {
        errno = EPROTO;
        goto ppm_abort;
    }

    /* Here we are... next byte begins the 24-bit data */
    filemap->image = p + 1;
    filemap_close(filemap);
    filemap = 0x0;
    return 0;

ppm_abort:
    filemap_close(filemap);
    filemap = 0x0;
    return -1;
}

static void print(struct world *world){
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
               X(world, b), Y(world, b), XF(world, b), YF(world, b), XV(world, b), YV(world, b));
    }
}


/*  Main program...
*/

int main(int argc, char **argv){
    int b;
    int steps;
    double rtime;
    struct timeval start;
    struct timeval end;
    struct filemap image_map;
    int dim_to_send[2];
            
    // Setting up MPI environment
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Setting up OpenMP Context
    omp_set_num_threads(2);

    struct world *world = (struct world*)calloc(1, sizeof *world);
    if (world == NULL) {
        fprintf(stderr, "Cannot calloc(world)\n");
        exit(1);
    }

    /* Get Parameters */
    if (argc != 5) {
        fprintf(stderr, "Usage: %s num_bodies secs_per_update ppm_output_file steps\n",
                argv[0]);
        exit(1);
    }
    if ((world->bodyCt = atol(argv[1])) > MAXBODIES ) {
        fprintf(stderr, "Using only %d bodies...\n", MAXBODIES);
        world->bodyCt = MAXBODIES;
    } else if (world->bodyCt < 2) {
        fprintf(stderr, "Using two bodies...\n");
        world->bodyCt = 2;
    }
    
    chunk_size = ceil((double)world->bodyCt / P);
    lborder = world_rank * chunk_size;
    rborder = MIN(lborder + chunk_size, world->bodyCt);

    // Allocate once
    forces = (double*)malloc(sizeof(double) * world->bodyCt * 2);
    forces2 = (double*)malloc(sizeof(double) * world->bodyCt * 2);
    coords = (double*)malloc(sizeof(double) * chunk_size * 2);
    coords2 = (double*)malloc(sizeof(double) * chunk_size * 2);

    if(on_master()){ // only master reads the file, then sends values to everybody...
        if (map_P6(argv[3], &world->xdim, &world->ydim, &image_map) == -1) {
            dim_to_send[0] = dim_to_send[1] = -1;
            fprintf(stderr, "Master cannot read %s: %s\n", argv[3], strerror(errno));
            MPI_Bcast(dim_to_send, 2, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            exit(1);
        }
        dim_to_send[0] = world->xdim; dim_to_send[1] = world->ydim;
    }

    MPI_Bcast(dim_to_send, 2, MPI_INT, 0, MPI_COMM_WORLD);
    if(!on_master()){
        world->xdim = dim_to_send[0]; world->ydim = dim_to_send[1];
    }
    
    steps = atoi(argv[4]);

    if(on_master()) fprintf(stderr, "Running N-body with %i bodies and %i steps\n", world->bodyCt, steps);

    /* Initialize simulation data */
    srand(SEED);
    for (b = 0; b < world->bodyCt; ++b) {
        X(world, b) = (rand() % world->xdim);
        Y(world, b) = (rand() % world->ydim);
        R(world, b) = 1 + ((b*b + 1.0) * sqrt(1.0 * ((world->xdim * world->xdim) + (world->ydim * world->ydim)))) /
                (25.0 * (world->bodyCt*world->bodyCt + 1.0));
        M(world, b) = R(world, b) * R(world, b) * R(world, b);
        XV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
        YV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
    }
    if(on_master())
        if (gettimeofday(&start, 0) != 0) {
            fprintf(stderr, "could not do timing\n");
            exit(1);
        }

    /* Main Loop */
    while (steps--) {
        clear_forces(world);
        gather_coords(world);
        compute_forces(world);
        gather_forces_reduce(world);
        compute_velocities(world);
        compute_positions(world);

        /* Flip old & new coordinates */
        world->old ^= 1;

    }
    MPI_Barrier(MPI_COMM_WORLD); // every node has to finish the job at this point
    if(on_master()){
        if (gettimeofday(&end, 0) != 0) {
            fprintf(stderr, "could not do timing\n");
            exit(1);
        }

        // Gather info from all process ...

        rtime = (end.tv_sec + (end.tv_usec / 1000000.0)) - 
                    (start.tv_sec + (start.tv_usec / 1000000.0));

        fprintf(stderr, "N-body took %10.3f seconds\n", rtime);
        for(int i = 1; i < P; ++i){
            MPI_Recv(&world->bodies[i * chunk_size], chunk_size * sizeof(struct bodyType) / sizeof(double), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);    
        }
        print(world);
    } else {
        // send info to master node ...
        MPI_Send(&world->bodies[lborder], chunk_size * sizeof(struct bodyType) / sizeof(double), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    free(world);
    free(forces);
    free(forces2);
    free(coords);
    free(coords2);
    return 0;
}
