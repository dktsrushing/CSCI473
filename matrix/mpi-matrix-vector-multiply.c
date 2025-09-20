/*
 * mpi_matrix-vector-multiply.c
 *
 * Usage:
 *   mpirun -np <P> ./mpi_matrix-vector-multiply <input A> <input B> <output C>
 *
 * Binary format (as in make-matrix):
 *   [int rows][int cols][double payload row-major]
 *
 * Timings (rank 0 prints):
 *   TIMING total_s=... read_s=... compute_s=... write_s=... m=... n=... p=...
 *     - read_s   : rank 0's time for reading A/B + Bcast(B) + Scatterv(A)
 *     - compute_s: MAX over ranks of local compute time (critical path)
 *     - write_s  : rank 0's time for Gatherv(C) + writing C
 *     - total_s  : overall wall time on rank 0
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <sys/time.h>

/* ---------------- one-malloc matrix container ---------------- */
typedef struct {
    size_t rows, cols;
    double **row;   /* row pointers */
    double  *data;  /* contiguous payload */
    void    *block; /* base allocation to free */
} Matrix;

/* ---------------- helpers: usage & errors ---------------- */
static void usage(const char *prog) {
    if (!prog) prog = "mpi_matrix-vector-multiply";
    fprintf(stderr,
        "Usage:\n"
        "  mpirun -np <P> %s <input A> <input B> <output C>\n",
        prog);
}

/* size_t overflow helpers */
static int mul_size_t(size_t a, size_t b, size_t *out) {
#if defined(__has_builtin)
#  if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, out);
#  endif
#endif
    if (a == 0 || b == 0) { *out = 0; return 0; }
    if (a > SIZE_MAX / b) return 1;
    *out = a * b;
    return 0;
}
static int add_size_t(size_t a, size_t b, size_t *out) {
#if defined(__has_builtin)
#  if __has_builtin(__builtin_add_overflow)
    return __builtin_add_overflow(a, b, out);
#  endif
#endif
    if (b > SIZE_MAX - a) return 1;
    *out = a + b;
    return 0;
}

/* Build one-malloc 2D layout (uninitialized payload) */
static int alloc_matrix(size_t rows, size_t cols, Matrix *M) {
    memset(M, 0, sizeof(*M));
    size_t ptrs_bytes = 0, n_elems = 0, payload_bytes = 0, total_bytes = 0;

    if (mul_size_t(rows, sizeof(double*), &ptrs_bytes) ||
        mul_size_t(rows, cols, &n_elems) ||
        mul_size_t(n_elems, sizeof(double), &payload_bytes) ||
        add_size_t(ptrs_bytes, payload_bytes, &total_bytes)) {
        return -1;
    }
    void *block = malloc(total_bytes);
    if (!block) return -1;

    double **row_ptrs = (double**)block;
    double  *payload  = (double*)((unsigned char*)block + ptrs_bytes);

    for (size_t r = 0; r < rows; ++r) {
        row_ptrs[r] = payload + r * cols;
    }

    M->rows  = rows;
    M->cols  = cols;
    M->row   = row_ptrs;
    M->data  = payload;
    M->block = block;
    return 0;
}

/* Rank 0: read matrix from file into one-malloc layout */
static int read_matrix_rank0(const char *path, Matrix *M) {
    memset(M, 0, sizeof(*M));
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Rank0: open '%s' failed: %s\n", path, strerror(errno));
        return -1;
    }
    int irows = 0, icols = 0;
    if (fread(&irows, sizeof(int), 1, fp) != 1 ||
        fread(&icols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Rank0: read header from '%s' failed: %s\n",
                path, ferror(fp) ? strerror(errno) : "unexpected EOF");
        fclose(fp);
        return -1;
    }
    if (irows <= 0 || icols <= 0) {
        fprintf(stderr, "Rank0: invalid dims in '%s': rows=%d cols=%d\n", path, irows, icols);
        fclose(fp);
        return -1;
    }
    size_t rows = (size_t)irows, cols = (size_t)icols;

    if (alloc_matrix(rows, cols, M) != 0) {
        fprintf(stderr, "Rank0: alloc_matrix(%zu,%zu) failed for '%s'\n", rows, cols, path);
        fclose(fp);
        return -1;
    }
    size_t n_elems = rows * cols;
    size_t nread = fread(M->data, sizeof(double), n_elems, fp);
    if (nread != n_elems) {
        fprintf(stderr, "Rank0: short read on '%s': expected %zu doubles, got %zu: %s\n",
                path, n_elems, nread, ferror(fp) ? strerror(errno) : "unexpected EOF");
        fclose(fp);
        free(M->block); memset(M, 0, sizeof(*M));
        return -1;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "Rank0: warning: fclose('%s') failed: %s\n", path, strerror(errno));
    }
    return 0;
}

/* Rank 0: write matrix to file from one-malloc layout */
static int write_matrix_rank0(const char *path, const Matrix *M) {
    if (M->rows > (size_t)INT_MAX || M->cols > (size_t)INT_MAX) {
        fprintf(stderr, "Rank0: dims exceed INT_MAX for header: %zu x %zu\n", M->rows, M->cols);
        return -1;
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Rank0: open '%s' for writing failed: %s\n", path, strerror(errno));
        return -1;
    }
    int irows = (int)M->rows;
    int icols = (int)M->cols;
    if (fwrite(&irows, sizeof(int), 1, fp) != 1 ||
        fwrite(&icols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Rank0: writing header to '%s' failed: %s\n", path, strerror(errno));
        fclose(fp);
        return -1;
    }
    size_t n_elems = M->rows * M->cols;
    size_t wrote = fwrite(M->data, sizeof(double), n_elems, fp);
    if (wrote != n_elems) {
        fprintf(stderr, "Rank0: short write to '%s': expected %zu doubles, wrote %zu: %s\n",
                path, n_elems, wrote, strerror(errno));
        fclose(fp);
        return -1;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "Rank0: warning: fclose('%s') failed: %s\n", path, strerror(errno));
    }
    return 0;
}

/* Row partitioning */
static void partition_rows(int m, int world_size, int *counts_rows, int *displs_rows) {
    int base = m / world_size;
    int rem  = m % world_size;
    int disp = 0;
    for (int r = 0; r < world_size; ++r) {
        int rows_r = base + (r < rem ? 1 : 0);
        counts_rows[r] = rows_r;
        displs_rows[r] = disp;
        disp += rows_r;
    }
}

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    /* Graceful usage failure when args are missing */
    if (argc != 4) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *pathA = argv[1];
    const char *pathB = argv[2];
    const char *pathC = argv[3];

    Matrix A_full = {0}, B_full = {0}; /* rank 0 only */
    Matrix B = {0};                    /* all ranks hold B (n x 1) */
    int m = 0, n = 0;

    /* Timing variables */
    double t_total_start = 0.0, t_total_end = 0.0;
    double t_read_start = 0.0, t_read_end = 0.0;       /* rank 0: read & distribute */
    double t_compute_start = 0.0, t_compute_end = 0.0; /* each rank local; reduce max */
    double t_write_start = 0.0, t_write_end = 0.0;     /* rank 0: gather+write */
    double compute_local = 0.0, compute_max = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    t_total_start = MPI_Wtime();

    /* ---------------- Rank 0: Read & distribute ---------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_read_start = MPI_Wtime();

    if (rank == 0) {
        if (read_matrix_rank0(pathA, &A_full) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (read_matrix_rank0(pathB, &B_full) != 0) {
            free(A_full.block);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if ((int)B_full.cols != 1) {
            fprintf(stderr, "Rank0: Error: B must be n x 1, but is %zu x %zu\n", B_full.rows, B_full.cols);
            free(A_full.block); free(B_full.block);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (A_full.cols != B_full.rows) {
            fprintf(stderr, "Rank0: Error: A(%zu x %zu) and B(%zu x %zu) mismatch (need A.cols==B.rows)\n",
                    A_full.rows, A_full.cols, B_full.rows, B_full.cols);
            free(A_full.block); free(B_full.block);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (A_full.rows > (size_t)INT_MAX || A_full.cols > (size_t)INT_MAX) {
            fprintf(stderr, "Rank0: Error: dims exceed INT_MAX.\n");
            free(A_full.block); free(B_full.block);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        m = (int)A_full.rows;
        n = (int)A_full.cols;
    }

    /* Broadcast dims to all ranks */
    if (MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS ||
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "MPI_Bcast of dims failed.\n");
        if (rank == 0) { free(A_full.block); free(B_full.block); }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (m <= 0 || n <= 0) {
        if (rank == 0) fprintf(stderr, "Invalid dims broadcast: m=%d n=%d\n", m, n);
        if (rank == 0) { free(A_full.block); free(B_full.block); }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Allocate B on all ranks and broadcast its payload */
    if (alloc_matrix((size_t)n, 1, &B) != 0) {
        if (rank == 0) { free(A_full.block); free(B_full.block); }
        fprintf(stderr, "Rank %d: alloc_matrix for B(%d x 1) failed.\n", rank, n);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (rank == 0) {
        memcpy(B.data, B_full.data, (size_t)n * sizeof(double));
    }
    if (MPI_Bcast(B.data, n, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: MPI_Bcast for B failed.\n", rank);
        if (rank == 0) { free(A_full.block); free(B_full.block); }
        free(B.block);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Partition rows and build counts/displs (rows, and in doubles for A/C payloads) */
    int *counts_rows = (int*)malloc((size_t)world * sizeof(int));
    int *displs_rows = (int*)malloc((size_t)world * sizeof(int));
    if (!counts_rows || !displs_rows) {
        fprintf(stderr, "Rank %d: malloc counts/displs failed.\n", rank);
        if (rank == 0) { free(A_full.block); free(B_full.block); }
        free(B.block);
        free(counts_rows); free(displs_rows);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    partition_rows(m, world, counts_rows, displs_rows);

    int *sendcounts_A = (int*)malloc((size_t)world * sizeof(int));
    int *displs_A     = (int*)malloc((size_t)world * sizeof(int));
    if (!sendcounts_A || !displs_A) {
        fprintf(stderr, "Rank %d: malloc sendcounts/displs for A failed.\n", rank);
        if (rank == 0) { free(A_full.block); free(B_full.block); }
        free(B.block);
        free(counts_rows); free(displs_rows);
        free(sendcounts_A); free(displs_A);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    for (int r = 0; r < world; ++r) {
        long elems = (long)counts_rows[r] * (long)n;
        long disp  = (long)displs_rows[r] * (long)n;
        if (elems > INT_MAX || disp > INT_MAX) {
            if (rank == 0) fprintf(stderr, "Error: message size exceeds MPI int on rank %d.\n", r);
            if (rank == 0) { free(A_full.block); free(B_full.block); }
            free(B.block);
            free(counts_rows); free(displs_rows);
            free(sendcounts_A); free(displs_A);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        sendcounts_A[r] = (int)elems;
        displs_A[r]     = (int)disp;
    }

    /* Allocate local A (only as big as needed) and receive via Scatterv */
    int local_rows = counts_rows[rank];
    Matrix A_local = {0};
    if (alloc_matrix((size_t)local_rows, (size_t)n, &A_local) != 0) {
        fprintf(stderr, "Rank %d: alloc_matrix A_local(%d x %d) failed.\n", rank, local_rows, n);
        if (rank == 0) { free(A_full.block); free(B_full.block); }
        free(B.block);
        free(counts_rows); free(displs_rows);
        free(sendcounts_A); free(displs_A);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const double *A_sendbuf = NULL;
    if (rank == 0) A_sendbuf = A_full.data;

    if (MPI_Scatterv(A_sendbuf, sendcounts_A, displs_A, MPI_DOUBLE,
                     A_local.data, (int)((long)local_rows*(long)n), MPI_DOUBLE,
                     0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: MPI_Scatterv of A failed.\n", rank);
        if (rank == 0) { free(A_full.block); free(B_full.block); }
        free(B.block);
        free(A_local.block);
        free(counts_rows); free(displs_rows);
        free(sendcounts_A); free(displs_A);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        /* Free A_full now that rows are distributed */
        free(A_full.block);
        A_full.block = NULL;
        memset(&A_full, 0, sizeof(A_full));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_read_end = MPI_Wtime();

    /* ---------------- Compute phase (each rank) ---------------- */
    Matrix C_local = {0};
    if (alloc_matrix((size_t)local_rows, 1, &C_local) != 0) {
        fprintf(stderr, "Rank %d: alloc_matrix C_local(%d x 1) failed.\n", rank, local_rows);
        if (rank == 0) { free(B_full.block); }
        free(B.block);
        free(A_local.block);
        free(counts_rows); free(displs_rows);
        free(sendcounts_A); free(displs_A);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_compute_start = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i) {
        const double *Ai = A_local.row[i];
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += Ai[k] * B.row[k][0];
        }
        C_local.row[i][0] = sum;
    }

    t_compute_end = MPI_Wtime();
    compute_local = t_compute_end - t_compute_start;

    /* Reduce compute to the max across ranks (critical path) */
    if (MPI_Reduce(&compute_local, &compute_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: MPI_Reduce for compute_max failed.\n", rank);
        if (rank == 0) { free(B_full.block); }
        free(B.block);
        free(C_local.block);
        free(A_local.block);
        free(counts_rows); free(displs_rows);
        free(sendcounts_A); free(displs_A);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* ---------------- Gather+Write phase ---------------- */
    int *recvcounts_C = NULL;
    int *displs_C = NULL;
    double *C_recvbuf = NULL;

    if (rank == 0) {
        t_write_start = MPI_Wtime();
        recvcounts_C = (int*)malloc((size_t)world * sizeof(int));
        displs_C     = (int*)malloc((size_t)world * sizeof(int));
        if (!recvcounts_C || !displs_C) {
            fprintf(stderr, "Rank0: malloc recvcounts/displs for C failed.\n");
            free(B.block);
            free(C_local.block);
            free(A_local.block);
            free(counts_rows); free(displs_rows);
            free(sendcounts_A); free(displs_A);
            free(recvcounts_C); free(displs_C);
            free(B_full.block);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int r = 0; r < world; ++r) {
            recvcounts_C[r] = counts_rows[r];
            displs_C[r]     = displs_rows[r];
        }
        C_recvbuf = (double*)malloc((size_t)m * sizeof(double));
        if (!C_recvbuf) {
            fprintf(stderr, "Rank0: malloc C_recvbuf(m=%d) failed.\n", m);
            free(B.block);
            free(C_local.block);
            free(A_local.block);
            free(counts_rows); free(displs_rows);
            free(sendcounts_A); free(displs_A);
            free(recvcounts_C); free(displs_C);
            free(B_full.block);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (MPI_Gatherv(C_local.data, local_rows, MPI_DOUBLE,
                    C_recvbuf, recvcounts_C, displs_C, MPI_DOUBLE,
                    0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: MPI_Gatherv of C failed.\n", rank);
        if (rank == 0) {
            free(C_recvbuf);
            free(recvcounts_C); free(displs_C);
        }
        free(B.block);
        free(C_local.block);
        free(A_local.block);
        free(counts_rows); free(displs_rows);
        free(sendcounts_A); free(displs_A);
        if (rank == 0) free(B_full.block);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        /* Package full C and write to disk */
        Matrix C_full = {0};
        if (alloc_matrix((size_t)m, 1, &C_full) != 0) {
            fprintf(stderr, "Rank0: alloc_matrix C_full(%d x 1) failed.\n", m);
            free(C_recvbuf);
            free(recvcounts_C); free(displs_C);
            free(B.block);
            free(C_local.block);
            free(A_local.block);
            free(counts_rows); free(displs_rows);
            free(sendcounts_A); free(displs_A);
            free(B_full.block);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memcpy(C_full.data, C_recvbuf, (size_t)m * sizeof(double));
        if (write_matrix_rank0(pathC, &C_full) != 0) {
            fprintf(stderr, "Rank0: write_matrix('%s') failed.\n", pathC);
            free(C_full.block);
            free(C_recvbuf);
            free(recvcounts_C); free(displs_C);
            free(B.block);
            free(C_local.block);
            free(A_local.block);
            free(counts_rows); free(displs_rows);
            free(sendcounts_A); free(displs_A);
            free(B_full.block);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        free(C_full.block);
        free(C_recvbuf);
        free(recvcounts_C); free(displs_C);

        t_write_end = MPI_Wtime();
    }

    /* ---------------- Finalize timings & print (rank 0) ---------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    t_total_end = MPI_Wtime();

    if (rank == 0) {
        double read_s    = t_read_end   - t_read_start;
        double compute_s = compute_max;
        double write_s   = t_write_end  - t_write_start;
        double total_s   = t_total_end  - t_total_start;

        /* Machine-readable one-liner */
        printf("TIMING total_s=%.9f read_s=%.9f compute_s=%.9f write_s=%.9f m=%d n=%d p=%d\n",
               total_s, read_s, compute_s, write_s, m, n, world);

        /* Human-readable breakdown */
        fprintf(stdout,
                "MPI Matrix-Vector Multiply (p=%d): A(%d x %d) * B(%d x 1) -> C(%d x 1)\n"
                "Elapsed times (seconds):\n"
                "  read (I/O + dist): %.9f\n"
                "  compute (max):     %.9f\n"
                "  write (gather+I/O):%.9f\n"
                "  total:             %.9f\n",
                world, m, n, n, m,
                read_s, compute_s, write_s, total_s);
        fflush(stdout);
    }

    /* ---------------- Cleanup ---------------- */
    if (rank == 0) {
        free(B_full.block);
    }
    free(B.block);
    free(C_local.block);
    free(A_local.block);

    free(counts_rows); free(displs_rows);
    free(sendcounts_A); free(displs_A);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

