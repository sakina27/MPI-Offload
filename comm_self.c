// comm_self.c: extra thread sits in MPI_Recv on MPI_COMM_SELF.

#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define N_LOCAL  1000000
#define NSTEPS   50
#define HALO_SIZE 10000

static void do_compute(double *a, int n) {
    for (int rep = 0; rep < 20; rep++) {
        for (int i = 1; i <= n; i++) {
            a[i] = 0.3333 * (a[i-1] + a[i] + a[i+1]);
        }
    }
}

void *self_comm_thread(void *arg) {
    MPI_Comm self_comm;
    MPI_Comm_dup(MPI_COMM_SELF, &self_comm);
    int dummy;
    MPI_Status st;

    // blocking recv that never completes (no send is posted)
    MPI_Recv(&dummy, 1, MPI_INT, 0, 123, self_comm, &st);

    MPI_Comm_free(&self_comm);
    return NULL;
}

int main(int argc, char **argv) {
    int provided;
    // two threads call MPI: main + self_comm_thread
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Run with at least 2 ranks\n");
        MPI_Finalize();
        return 0;
    }

    // launch the COMM_SELF progress thread
    pthread_t tid;
    pthread_create(&tid, NULL, self_comm_thread, NULL);

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    double *a = malloc((N_LOCAL + 2) * sizeof(double));
    for (int i = 0; i < N_LOCAL + 2; i++) a[i] = rank;

    double total_iter = 0.0, total_comp = 0.0;

    for (int step = 0; step < NSTEPS; step++) {
        MPI_Request reqs[4];
        MPI_Status stats[4];

        double t_iter_start = MPI_Wtime();

        MPI_Irecv(&a[0],         HALO_SIZE, MPI_DOUBLE, left,  0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&a[N_LOCAL+1], HALO_SIZE, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(&a[1],         HALO_SIZE, MPI_DOUBLE, left,  1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&a[N_LOCAL],   HALO_SIZE, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &reqs[3]);

        double t_comp_start = MPI_Wtime();
        do_compute(a, N_LOCAL);
        double t_comp_end   = MPI_Wtime();

        MPI_Waitall(4, reqs, stats);

        double t_iter_end = MPI_Wtime();

        total_comp += t_comp_end  - t_comp_start;
        total_iter += t_iter_end  - t_iter_start;
    }

    double avg_comp = total_comp / NSTEPS;
    double avg_iter = total_iter / NSTEPS;
    double avg_over = avg_iter - avg_comp;

    double g_comp, g_iter, g_over;
    MPI_Reduce(&avg_comp, &g_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_iter, &g_iter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_over, &g_over, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("=== COMM_SELF ===\n");
        printf("avg compute time per step     = %.6f s\n", g_comp);
        printf("avg total step time           = %.6f s\n", g_iter);
        printf("avg comm overhead             = %.6f s\n", g_over);
        printf("overlap fraction (rough)      = %.2f %%\n",
               100.0 * (1.0 - g_over / (g_iter)));
    }

    // terminate the COMM_SELF thread (hacky)
    pthread_cancel(tid);
    pthread_join(tid, NULL);

    free(a);
    MPI_Finalize();
    return 0;
}
