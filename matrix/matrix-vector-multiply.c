/*
 * matrix-vector-multiply.c
 *
 * Usage:
 *   ./matrix-vector-multiply <input matrix A> <input vector B> <output C>
 *
 * File format (same as make-matrix): [int rows][int cols][double payload row-major]
 * Timing:
 *   - Prints machine-readable one-line summary:
 *       TIMING total_s=<..> read_s=<..> compute_s=<..> write_s=<..> m=<..> n=<..>
 *   - Also prints a human-readable breakdown.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>   /* INT_MAX */
#include <stdint.h>
#include <time.h>     /* clock_gettime */
#include <sys/time.h>

typedef struct {
    size_t rows, cols;
    double **row;  /* row pointers */
    double  *data; /* contiguous payload */
    void    *block;
} Matrix;

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s <input matrix A> <input vector B> <output C>\n"
        "  - A: m x n, B: n x 1, C: m x 1\n", prog);
}

/* -------- timing helpers (monotonic clock) -------- */

static double now_sec(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    /* Very old fallback; not expected on modern macOS/Linux */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
#endif
}

/* -------- size_t overflow helpers -------- */

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

/* -------- I/O helpers (one-malloc layout) -------- */

static int read_matrix(const char *path, Matrix *M_out) {
    memset(M_out, 0, sizeof(*M_out));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: open '%s' failed: %s\n", path, strerror(errno));
        return -1;
    }

    int irows = 0, icols = 0;
    if (fread(&irows, sizeof(int), 1, fp) != 1 ||
        fread(&icols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error: reading header from '%s': %s\n",
                path, ferror(fp) ? strerror(errno) : "unexpected EOF");
        fclose(fp);
        return -1;
    }
    if (irows <= 0 || icols <= 0) {
        fprintf(stderr, "Error: invalid dims in '%s': rows=%d cols=%d\n", path, irows, icols);
        fclose(fp);
        return -1;
    }

    size_t rows = (size_t)irows, cols = (size_t)icols;

    size_t n_elems = 0, ptrs_bytes = 0, payload_bytes = 0, total_bytes = 0;
    if (mul_size_t(rows, cols, &n_elems)) {
        fprintf(stderr, "Error: rows*cols overflow for '%s'\n", path);
        fclose(fp);
        return -1;
    }
    if (mul_size_t(rows, sizeof(double*), &ptrs_bytes)) {
        fprintf(stderr, "Error: pointer table size overflow for '%s'\n", path);
        fclose(fp);
        return -1;
    }
    if (mul_size_t(n_elems, sizeof(double), &payload_bytes)) {
        fprintf(stderr, "Error: payload size overflow for '%s'\n", path);
        fclose(fp);
        return -1;
    }
    if (add_size_t(ptrs_bytes, payload_bytes, &total_bytes)) {
        fprintf(stderr, "Error: total allocation size overflow for '%s'\n", path);
        fclose(fp);
        return -1;
    }

    void *block = malloc(total_bytes);
    if (!block) {
        fprintf(stderr, "Error: malloc(%zu) failed for '%s': %s\n",
                total_bytes, path, strerror(errno));
        fclose(fp);
        return -1;
    }

    double **row_ptrs = (double**)block;
    double  *payload  = (double*)((unsigned char*)block + ptrs_bytes);

    for (size_t r = 0; r < rows; ++r) {
        row_ptrs[r] = payload + r * cols;
    }

    size_t nread = fread(payload, sizeof(double), n_elems, fp);
    if (nread != n_elems) {
        fprintf(stderr, "Error: short read on '%s': expected %zu doubles, got %zu: %s\n",
                path, n_elems, nread, ferror(fp) ? strerror(errno) : "unexpected EOF");
        free(block);
        fclose(fp);
        return -1;
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "Warning: fclose('%s') failed: %s\n", path, strerror(errno));
    }

    M_out->rows = rows;
    M_out->cols = cols;
    M_out->row  = row_ptrs;
    M_out->data = payload;
    M_out->block= block;
    return 0;
}

static int write_matrix(const char *path, const Matrix *M) {
    if (M->rows > (size_t)INT_MAX || M->cols > (size_t)INT_MAX) {
        fprintf(stderr, "Error: dims exceed INT_MAX for header: %zu x %zu\n", M->rows, M->cols);
        return -1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: open '%s' for writing failed: %s\n", path, strerror(errno));
        return -1;
    }

    int irows = (int)M->rows;
    int icols = (int)M->cols;

    if (fwrite(&irows, sizeof(int), 1, fp) != 1 ||
        fwrite(&icols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error: writing header to '%s' failed: %s\n", path, strerror(errno));
        fclose(fp);
        return -1;
    }

    size_t n_elems = 0;
    if (mul_size_t(M->rows, M->cols, &n_elems)) {
        fprintf(stderr, "Error: rows*cols overflow on write\n");
        fclose(fp);
        return -1;
    }

    size_t wrote = fwrite(M->data, sizeof(double), n_elems, fp);
    if (wrote != n_elems) {
        fprintf(stderr, "Error: short write to '%s': expected %zu doubles, wrote %zu: %s\n",
                path, n_elems, wrote, strerror(errno));
        fclose(fp);
        return -1;
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "Warning: fclose('%s') failed: %s\n", path, strerror(errno));
    }
    return 0;
}

/* ---------------- main (with timing) ---------------- */

int main(int argc, char **argv) {
    if (argc != 4) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const double t_start = now_sec();

    const char *pathA = argv[1];
    const char *pathB = argv[2];
    const char *pathC = argv[3];

    Matrix A = {0}, B = {0}, C = {0};

    /* ----- read timing ----- */
    const double t_read_start = now_sec();

    if (read_matrix(pathA, &A) != 0) {
        return EXIT_FAILURE;
    }
    if (read_matrix(pathB, &B) != 0) {
        free(A.block);
        return EXIT_FAILURE;
    }

    const double t_after_read = now_sec();
    const double read_s = t_after_read - t_read_start;

    /* Dim checks */
    if (B.cols != 1) {
        fprintf(stderr, "Error: Input B must be n x 1, but is %zu x %zu\n", B.rows, B.cols);
        free(A.block); free(B.block);
        return EXIT_FAILURE;
    }
    if (A.cols != B.rows) {
        fprintf(stderr, "Error: Dimension mismatch: A=%zu x %zu, B=%zu x %zu (need A.cols==B.rows)\n",
                A.rows, A.cols, B.rows, B.cols);
        free(A.block); free(B.block);
        return EXIT_FAILURE;
    }

    /* Allocate C: m x 1 with one malloc() */
    C.rows = A.rows;
    C.cols = 1;

    size_t ptrs_bytes = 0, n_elems_C = 0, payload_bytes = 0, total_bytes = 0;
    if (mul_size_t(C.rows, sizeof(double*), &ptrs_bytes) ||
        mul_size_t(C.rows, C.cols, &n_elems_C) ||
        mul_size_t(n_elems_C, sizeof(double), &payload_bytes) ||
        add_size_t(ptrs_bytes, payload_bytes, &total_bytes)) {
        fprintf(stderr, "Error: overflow sizing C\n");
        free(A.block); free(B.block);
        return EXIT_FAILURE;
    }

    C.block = malloc(total_bytes);
    if (!C.block) {
        fprintf(stderr, "Error: malloc(%zu) for C failed: %s\n", total_bytes, strerror(errno));
        free(A.block); free(B.block);
        return EXIT_FAILURE;
    }

    C.row  = (double**)C.block;
    C.data = (double*)((unsigned char*)C.block + ptrs_bytes);
    for (size_t r = 0; r < C.rows; ++r) {
        C.row[r] = C.data + r * C.cols; /* cols == 1 */
    }

    /* ----- compute timing ----- */
    const double t_compute_start = now_sec();

    /* C = A * B (O(m*n)) */
    const size_t m = A.rows;
    const size_t n = A.cols; /* == B.rows */
    for (size_t i = 0; i < m; ++i) {
        const double *Ai = A.row[i];
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum += Ai[k] * B.row[k][0];  /* B is n x 1 */
        }
        C.row[i][0] = sum;
    }

    const double t_after_compute = now_sec();
    const double compute_s = t_after_compute - t_compute_start;

    /* ----- write timing ----- */
    const double t_write_start = now_sec();

    if (write_matrix(pathC, &C) != 0) {
        free(A.block); free(B.block); free(C.block);
        return EXIT_FAILURE;
    }

    const double t_after_write = now_sec();
    const double write_s = t_after_write - t_write_start;

    /* Totals */
    const double total_s = now_sec() - t_start;

    /* Machine-readable one-liner for scripts */
    printf("TIMING total_s=%.9f read_s=%.9f compute_s=%.9f write_s=%.9f m=%zu n=%zu\n",
           total_s, read_s, compute_s, write_s, m, n);

    /* Human-readable breakdown */
    fprintf(stdout,
            "Matrix-Vector Multiply: A(%zu x %zu) * B(%zu x %zu) -> C(%zu x %zu)\n"
            "Elapsed times (seconds):\n"
            "  read:    %.9f\n"
            "  compute: %.9f\n"
            "  write:   %.9f\n"
            "  total:   %.9f\n",
            A.rows, A.cols, B.rows, B.cols, C.rows, C.cols,
            read_s, compute_s, write_s, total_s);

    free(A.block);
    free(B.block);
    free(C.block);
    return EXIT_SUCCESS;
}

