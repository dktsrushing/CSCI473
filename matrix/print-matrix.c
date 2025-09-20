/*
 * print-matrix.c
 *
 * Usage:
 *   ./print-matrix -i <input_file>
 *
 * Expects the binary format produced by make-matrix:
 *   [int rows][int cols][double data in row-major order]
 *
 * Behavior:
 *   - Reads header
 *   - Allocates a single block: [double* row_ptrs[rows]] + [double payload[rows*cols]]
 *   - Wires row pointers to the payload
 *   - Reads all doubles into the payload
 *   - Prints values in row-major order
 *
 * Notes:
 *   - Assumes host endianness and sizeof(int)/sizeof(double) match the writer.
 */
#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <limits.h>   /* INT_MAX */
#include <unistd.h>   /* getopt(), optarg */

static void
print_usage_and_exit(const char *prog, int code)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s -i <input_file>\n"
        "\nExample:\n"
        "  %s -i matrix.bin\n",
        prog, prog);
    exit(code);
}

/* Safe multiply for size_t with a fallback when compiler builtins aren't available */
static int
mul_size_t(size_t a, size_t b, size_t *out)
{
#if defined(__has_builtin)
#  if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, out);
#  endif
#endif
    if (a == 0 || b == 0) { *out = 0; return 0; }
    if (a > SIZE_MAX / b) return 1; /* overflow */
    *out = a * b;
    return 0;
}

int
main(int argc, char **argv)
{
    const char *prog = argv[0];
    const char *in_path = NULL;

    if (argc == 1) {
        print_usage_and_exit(prog, EXIT_FAILURE);
    }

    int opt;
    while ((opt = getopt(argc, argv, "i:h")) != -1) {
        switch (opt) {
            case 'i':
                in_path = optarg;
                break;
            case 'h':
                print_usage_and_exit(prog, EXIT_SUCCESS);
                break;
            default:
                print_usage_and_exit(prog, EXIT_FAILURE);
        }
    }

    if (!in_path) {
        fprintf(stderr, "Error: -i <input_file> is required.\n");
        return EXIT_FAILURE;
    }

    /* Open file */
    FILE *fp = fopen(in_path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: failed to open '%s' for reading: %s\n",
                in_path, strerror(errno));
        return EXIT_FAILURE;
    }

    /* Read header: two ints */
    int irows = 0, icols = 0;
    if (fread(&irows, sizeof(int), 1, fp) != 1 ||
        fread(&icols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error: failed to read header from '%s': %s\n",
                in_path, ferror(fp) ? strerror(errno) : "unexpected EOF");
        fclose(fp);
        return EXIT_FAILURE;
    }

    if (irows <= 0 || icols <= 0) {
        fprintf(stderr, "Error: invalid header values rows=%d cols=%d\n", irows, icols);
        fclose(fp);
        return EXIT_FAILURE;
    }

    /* Cast to size_t and check */
    size_t n_rows = (size_t)irows;
    size_t n_cols = (size_t)icols;

    /* Compute sizes and allocate the same single block layout */
    size_t n_elems;
    if (mul_size_t(n_rows, n_cols, &n_elems)) {
        fprintf(stderr, "Error: rows*cols overflows size_t.\n");
        fclose(fp);
        return EXIT_FAILURE;
    }

    size_t ptrs_bytes;
    if (mul_size_t(n_rows, sizeof(double*), &ptrs_bytes)) {
        fprintf(stderr, "Error: pointer table size overflow.\n");
        fclose(fp);
        return EXIT_FAILURE;
    }

    size_t payload_bytes;
    if (mul_size_t(n_elems, sizeof(double), &payload_bytes)) {
        fprintf(stderr, "Error: payload size overflow.\n");
        fclose(fp);
        return EXIT_FAILURE;
    }

    size_t total_bytes = ptrs_bytes + payload_bytes;
    void *block = malloc(total_bytes);
    if (!block) {
        fprintf(stderr, "Error: malloc(%zu) failed: %s\n", total_bytes, strerror(errno));
        fclose(fp);
        return EXIT_FAILURE;
    }

    double **row_ptrs = (double **)block;
    double  *payload  = (double *)((unsigned char *)block + ptrs_bytes);

    /* Wire row pointers */
    for (size_t r = 0; r < n_rows; ++r) {
        row_ptrs[r] = payload + r * n_cols;
    }

    /* Read the full payload into memory */
    size_t nread = fread(payload, sizeof(double), n_elems, fp);
    if (nread != n_elems) {
        fprintf(stderr, "Error: short read: expected %zu doubles, got %zu: %s\n",
                n_elems, nread, ferror(fp) ? strerror(errno) : "unexpected EOF");
        free(block);
        fclose(fp);
        return EXIT_FAILURE;
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "Warning: fclose('%s') failed: %s\n", in_path, strerror(errno));
    }

    /* Print matrix in row-major order */
    printf("Matrix %zu x %zu from '%s'\n", n_rows, n_cols, in_path);
    for (size_t r = 0; r < n_rows; ++r) {
        const double *row = row_ptrs[r];
        for (size_t c = 0; c < n_cols; ++c) {
            /* Use %.17g to preserve double precision while keeping output readable */
            printf("% .17g%s", row[c], (c + 1 == n_cols) ? "" : " ");
        }
        putchar('\n');
    }

    free(block);
    return EXIT_SUCCESS;
}

