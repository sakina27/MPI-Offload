// offload.c: Offload TEAM version
// Multiple offload threads perform all MPI; compute thread never calls MPI.
// Updated for HALO_SIZE-safe halo buffers.

#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define N_LOCAL   10000
#define NSTEPS    30
#define MAX_REQ   2048
#define QSIZE     2048
#define HALO_SIZE 2000000   // BIG halos to emphasize communication cost

// Number of offload threads ("Offload Team")
#define N_OFFLOAD_THREADS 2

int rank_global, size_global;

// ---------- request pool ----------

// All offload threads share this pool
MPI_Request req_pool[MAX_REQ];
volatile int done_flag[MAX_REQ];

// Free-list for request indices
int free_list[MAX_REQ];
int free_head;

// Protects free_list and req_pool modifications
pthread_mutex_t free_lock = PTHREAD_MUTEX_INITIALIZER;

// Protects MPI calls that touch req_pool[] and MPI_Testany
pthread_mutex_t req_lock  = PTHREAD_MUTEX_INITIALIZER;

void init_request_pool(void) {
    pthread_mutex_lock(&free_lock);
    for (int i = 0; i < MAX_REQ; i++) {
        free_list[i]  = i + 1;
        done_flag[i]  = 0;
        req_pool[i]   = MPI_REQUEST_NULL;
    }
    free_list[MAX_REQ - 1] = -1;
    free_head = 0;
    pthread_mutex_unlock(&free_lock);
}

int alloc_req_index(void) {
    pthread_mutex_lock(&free_lock);
    int idx = free_head;
    if (idx == -1) {
        pthread_mutex_unlock(&free_lock);
        return -1;  // pool exhausted
    }
    free_head = free_list[idx];
    done_flag[idx] = 0;
    req_pool[idx]  = MPI_REQUEST_NULL;
    pthread_mutex_unlock(&free_lock);
    return idx;
}

void free_req_index(int idx) {
    if (idx < 0 || idx >= MAX_REQ) return;
    pthread_mutex_lock(&free_lock);
    req_pool[idx] = MPI_REQUEST_NULL;
    free_list[idx] = free_head;
    free_head = idx;
    pthread_mutex_unlock(&free_lock);
}

// ---------- command queue ----------

typedef enum { CMD_ISEND, CMD_IRECV, CMD_SHUTDOWN } CmdType;

typedef struct {
    CmdType      type;
    void        *buf;
    int          count;
    MPI_Datatype dt;
    int          peer;
    int          tag;
    MPI_Comm     comm;
    int          req_index;
} Command;

// Single shared queue for all offload threads
Command queue[QSIZE];
int q_head = 0, q_tail = 0, q_count = 0;
pthread_mutex_t q_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  q_not_empty = PTHREAD_COND_INITIALIZER;
pthread_cond_t  q_not_full  = PTHREAD_COND_INITIALIZER;

// Enqueue from compute thread (blocking)
void enqueue_cmd(const Command *c) {
    pthread_mutex_lock(&q_lock);
    while (q_count == QSIZE)
        pthread_cond_wait(&q_not_full, &q_lock);

    queue[q_tail] = *c;
    q_tail = (q_tail + 1) % QSIZE;
    q_count++;

    pthread_cond_signal(&q_not_empty);
    pthread_mutex_unlock(&q_lock);
}

// Non-blocking dequeue from offload team threads
int try_dequeue_cmd(Command *c) {
    int have = 0;
    pthread_mutex_lock(&q_lock);
    if (q_count > 0) {
        *c = queue[q_head];
        q_head = (q_head + 1) % QSIZE;
        q_count--;
        pthread_cond_signal(&q_not_full);
        have = 1;
    }
    pthread_mutex_unlock(&q_lock);
    return have;
}

// ---------- offload team threads ----------

volatile int shutdown_flag = 0;

void *offload_thread_main(void *arg) {
    (void)arg;

    while (!shutdown_flag) {
        Command cmd;
        int got = try_dequeue_cmd(&cmd);

        if (got) {
            if (cmd.type == CMD_SHUTDOWN) {
                // One thread sees shutdown; signal all to exit
                shutdown_flag = 1;
                continue;
            }

            // Issue the MPI call; protect with req_lock so only
            // one offload thread enters these MPI calls at a time.
            pthread_mutex_lock(&req_lock);
            if (cmd.type == CMD_ISEND) {
                MPI_Isend(cmd.buf, cmd.count, cmd.dt,
                          cmd.peer, cmd.tag, cmd.comm,
                          &req_pool[cmd.req_index]);
            } else if (cmd.type == CMD_IRECV) {
                MPI_Irecv(cmd.buf, cmd.count, cmd.dt,
                          cmd.peer, cmd.tag, cmd.comm,
                          &req_pool[cmd.req_index]);
            }
            pthread_mutex_unlock(&req_lock);
        }

        // Drive MPI progress continuously (check for completed requests)
        int index, flag;
        MPI_Status st;
        pthread_mutex_lock(&req_lock);
        MPI_Testany(MAX_REQ, req_pool, &index, &flag, &st);
        if (flag && index != MPI_UNDEFINED) {
            done_flag[index] = 1;
            // Note: we don't reset req_pool[index] here since
            // free_req_index() will clean it up later.
        }
        pthread_mutex_unlock(&req_lock);
    }

    return NULL;
}

// ---------- wrapper API used by compute thread ----------

static MPI_Request make_fake(int idx) { return (MPI_Request)(long)(idx + 1); }
static int get_index(MPI_Request r)    { return (int)((long)r) - 1; }

int my_Irecv(void *buf, int count, MPI_Datatype dt,
             int src, int tag, MPI_Comm comm,
             MPI_Request *req) {
    int idx = alloc_req_index();
    if (idx < 0) return MPI_ERR_OTHER;

    Command c = {
        .type = CMD_IRECV,
        .buf = buf,
        .count = count,
        .dt = dt,
        .peer = src,
        .tag = tag,
        .comm = comm,
        .req_index = idx
    };
    enqueue_cmd(&c);

    *req = make_fake(idx);
    return MPI_SUCCESS;
}

int my_Isend(void *buf, int count, MPI_Datatype dt,
             int dest, int tag, MPI_Comm comm,
             MPI_Request *req) {
    int idx = alloc_req_index();
    if (idx < 0) return MPI_ERR_OTHER;

    Command c = {
        .type = CMD_ISEND,
        .buf = buf,
        .count = count,
        .dt = dt,
        .peer = dest,
        .tag = tag,
        .comm = comm,
        .req_index = idx
    };
    enqueue_cmd(&c);

    *req = make_fake(idx);
    return MPI_SUCCESS;
}

int my_Wait(MPI_Request *req) {
    int idx = get_index(*req);
    // Busy-wait until some offload thread marks this request as done
    while (!done_flag[idx]) {
        // could add sched_yield() here if you want to be nicer to CPU
        sched_yield();
    }
    *req = MPI_REQUEST_NULL;
    free_req_index(idx);
    return MPI_SUCCESS;
}

// ---------- compute thread ----------

static void do_compute(double *a, int n) {
    for (int rep = 0; rep < 20; rep++) {
        for (int i = 1; i <= n; i++) {
            a[i] = 0.3333 * (a[i-1] + a[i] + a[i+1]);
        }
    }
}

void *compute_thread_main(void *arg) {
    (void)arg;

    int left  = (rank_global - 1 + size_global) % size_global;
    int right = (rank_global + 1) % size_global;

    double *a = malloc((N_LOCAL + 2) * sizeof(double));
    for (int i = 0; i < N_LOCAL + 2; i++) a[i] = rank_global;

    // SAFE HALO BUFFERS (no out-of-bounds when HALO_SIZE > 1)
    double *halo_left_send  = malloc(HALO_SIZE * sizeof(double));
    double *halo_right_send = malloc(HALO_SIZE * sizeof(double));
    double *halo_left_recv  = malloc(HALO_SIZE * sizeof(double));
    double *halo_right_recv = malloc(HALO_SIZE * sizeof(double));

    for (int i = 0; i < HALO_SIZE; i++) {
        halo_left_send[i]  = rank_global;
        halo_right_send[i] = rank_global;
    }

    double total_iter = 0.0, total_comp = 0.0;

    for (int step = 0; step < NSTEPS; step++) {
        MPI_Request r[4];

        double t_iter_start = MPI_Wtime();

        // Post halo exchanges via offload team
        my_Irecv(halo_left_recv,  HALO_SIZE, MPI_DOUBLE, left,  0, MPI_COMM_WORLD, &r[0]);
        my_Irecv(halo_right_recv, HALO_SIZE, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &r[1]);
        my_Isend(halo_left_send,  HALO_SIZE, MPI_DOUBLE, left,  1, MPI_COMM_WORLD, &r[2]);
        my_Isend(halo_right_send, HALO_SIZE, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &r[3]);

        double t_comp_start = MPI_Wtime();
        do_compute(a, N_LOCAL);
        double t_comp_end   = MPI_Wtime();

        for (int i = 0; i < 4; i++)
            my_Wait(&r[i]);

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

    if (rank_global == 0) {
        printf("=== OFFLOAD (TEAM) ===\n");
        printf("avg compute time per step     = %.6f s\n", g_comp);
        printf("avg total step time           = %.6f s\n", g_iter);
        printf("avg comm overhead             = %.6f s\n", g_over);
        printf("overlap fraction (rough)      = %.2f %%\n",
               100.0 * (1.0 - g_over / g_iter));
    }

    free(halo_left_send);
    free(halo_right_send);
    free(halo_left_recv);
    free(halo_right_recv);
    free(a);

    return NULL;
}

// ---------- main ----------

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);
    MPI_Comm_size(MPI_COMM_WORLD, &size_global);

    if (size_global < 2) {
        if (rank_global == 0)
            fprintf(stderr, "Run with at least 2 ranks\n");
        MPI_Finalize();
        return 0;
    }

    if (provided < MPI_THREAD_MULTIPLE && rank_global == 0) {
        fprintf(stderr,
                "Warning: MPI implementation does not truly support MPI_THREAD_MULTIPLE (provided=%d)\n",
                provided);
    }

    init_request_pool();

    pthread_t offload_tids[N_OFFLOAD_THREADS];
    pthread_t compute_tid;

    // Start offload team
    for (int i = 0; i < N_OFFLOAD_THREADS; i++) {
        pthread_create(&offload_tids[i], NULL, offload_thread_main, NULL);
    }

    // Start compute thread
    pthread_create(&compute_tid, NULL, compute_thread_main, NULL);

    // Wait for compute thread to finish all steps
    pthread_join(compute_tid, NULL);

    // Send a single shutdown command; one offload thread will set shutdown_flag=1
    Command c;
    c.type = CMD_SHUTDOWN;
    c.buf = NULL;
    c.count = 0;
    c.dt = MPI_DOUBLE;
    c.peer = 0;
    c.tag = 0;
    c.comm = MPI_COMM_WORLD;
    c.req_index = -1;
    enqueue_cmd(&c);

    // Wait for offload team to exit
    for (int i = 0; i < N_OFFLOAD_THREADS; i++) {
        pthread_join(offload_tids[i], NULL);
    }

    MPI_Finalize();
    return 0;
}
