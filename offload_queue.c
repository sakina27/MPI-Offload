// offload.c: Dedicated offload thread performs all MPI; compute thread never
// calls MPI directly. Updated for HALO_SIZE-safe halo buffers
// and SEPARATE QUEUES for small vs large messages.

#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define N_LOCAL   10000
#define NSTEPS    30
#define MAX_REQ   2048
#define QSIZE     2048
#define HALO_SIZE 2000000   // BIG halos to emphasize communication cost

// Simple threshold on "count" to separate small vs large messages.
// You can tune this later (e.g., 1024, 4096, etc).
#define SMALL_MSG_THRESHOLD  1024

int rank_global, size_global;

// ---------- request pool ----------

MPI_Request req_pool[MAX_REQ];
volatile int done_flag[MAX_REQ];
int free_list[MAX_REQ];
int free_head;
pthread_mutex_t free_lock = PTHREAD_MUTEX_INITIALIZER;

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
        return -1;
    }
    free_head = free_list[idx];
    done_flag[idx] = 0;
    req_pool[idx]  = MPI_REQUEST_NULL;
    pthread_mutex_unlock(&free_lock);
    return idx;
}

void free_req_index(int idx) {
    pthread_mutex_lock(&free_lock);
    req_pool[idx] = MPI_REQUEST_NULL;
    free_list[idx] = free_head;
    free_head = idx;
    pthread_mutex_unlock(&free_lock);
}

// ---------- command + two queues ----------

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

// Generic queue type
typedef struct {
    Command buf[QSIZE];
    int head;
    int tail;
    int count;
    pthread_mutex_t lock;
    pthread_cond_t  not_empty;
    pthread_cond_t  not_full;
} CommandQueue;

// One queue for small messages, one for large messages
static CommandQueue small_q = {
    .head = 0, .tail = 0, .count = 0,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .not_empty = PTHREAD_COND_INITIALIZER,
    .not_full  = PTHREAD_COND_INITIALIZER
};

static CommandQueue large_q = {
    .head = 0, .tail = 0, .count = 0,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .not_empty = PTHREAD_COND_INITIALIZER,
    .not_full  = PTHREAD_COND_INITIALIZER
};

// Helper to enqueue into a specific queue (blocking)
static void enqueue_to_queue(CommandQueue *q, const Command *c) {
    pthread_mutex_lock(&q->lock);
    while (q->count == QSIZE)
        pthread_cond_wait(&q->not_full, &q->lock);

    q->buf[q->tail] = *c;
    q->tail = (q->tail + 1) % QSIZE;
    q->count++;

    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}

// Helper to try-dequeue from a specific queue (non-blocking)
static int try_dequeue_from_queue(CommandQueue *q, Command *c) {
    int have = 0;
    pthread_mutex_lock(&q->lock);
    if (q->count > 0) {
        *c = q->buf[q->head];
        q->head = (q->head + 1) % QSIZE;
        q->count--;
        pthread_cond_signal(&q->not_full);
        have = 1;
    }
    pthread_mutex_unlock(&q->lock);
    return have;
}

// Public enqueue: pick small or large queue based on message size
void enqueue_cmd(const Command *c) {
    if (c->count <= SMALL_MSG_THRESHOLD) {
        // route small messages to small_q
        enqueue_to_queue(&small_q, c);
    } else {
        // large messages go to large_q
        enqueue_to_queue(&large_q, c);
    }
}

// Public dequeue: try small queue first, then large
int try_dequeue_cmd(Command *c) {
    // Always prioritize small messages
    if (try_dequeue_from_queue(&small_q, c))
        return 1;
    if (try_dequeue_from_queue(&large_q, c))
        return 1;
    return 0;
}

// ---------- offload thread ----------

volatile int shutdown_flag = 0;

void *offload_thread_main(void *arg) {
    (void)arg;

    while (!shutdown_flag) {
        Command cmd;
        int got = try_dequeue_cmd(&cmd);

        if (got) {
            if (cmd.type == CMD_SHUTDOWN) {
                shutdown_flag = 1;
                continue;
            }

            if (cmd.type == CMD_ISEND) {
                MPI_Isend(cmd.buf, cmd.count, cmd.dt,
                          cmd.peer, cmd.tag, cmd.comm,
                          &req_pool[cmd.req_index]);
            } else if (cmd.type == CMD_IRECV) {
                MPI_Irecv(cmd.buf, cmd.count, cmd.dt,
                          cmd.peer, cmd.tag, cmd.comm,
                          &req_pool[cmd.req_index]);
            }
        }

        // Drive MPI progress continuously
        int index, flag;
        MPI_Status st;
        MPI_Testany(MAX_REQ, req_pool, &index, &flag, &st);
        if (flag && index != MPI_UNDEFINED) {
            done_flag[index] = 1;
        }
    }

    return NULL;
}

// ---------- wrapper API ----------

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
    while (!done_flag[idx]) {
        // Optional: sched_yield() or tiny sleep to reduce CPU burn
        // sched_yield();
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

    // SAFE HALO BUFFERS
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
        printf("=== OFFLOAD (SEPARATE QUEUES) ===\n");
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

    init_request_pool();

    pthread_t offload_tid, compute_tid;
    pthread_create(&offload_tid, NULL, offload_thread_main, NULL);
    pthread_create(&compute_tid, NULL, compute_thread_main, NULL);

    pthread_join(compute_tid, NULL);

    // send shutdown command so offload thread exits cleanly
    Command c;
    c.type      = CMD_SHUTDOWN;
    c.buf       = NULL;
    c.count     = 0;
    c.dt        = MPI_DOUBLE;
    c.peer      = 0;
    c.tag       = 0;
    c.comm      = MPI_COMM_WORLD;
    c.req_index = -1;
    enqueue_cmd(&c);

    pthread_join(offload_tid, NULL);

    MPI_Finalize();
    return 0;
}
