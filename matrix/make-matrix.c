/*
 * make-matrix.c
 *
 * Usage (both forms supported):
 *   Long-ish (original):  ./make-matrix -rows <num_rows> -cols <num_cols> -l <lower> -u <upper> -o <file>
 *   Short (getopt):       ./make-matrix -r <num_rows>    -c <num_cols>    -l <lower> -u <upper> -o <file>
 *
 * Behavior:
 *   - Creates a rows x cols matrix of double-precision values drawn uniformly at random in [lower, upper].
 *   - Binary file layout:
 *       [int rows][int cols][double data in row-major order]
 *   - Single malloc() 2D layout: [double* row_ptrs[rows]] then payload of rows*cols doubles.
 *
 * Notes:
 *   - Uses host endianness/sizes. For portable on-disk format, add explicit endianness handling.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <limits.h>   /* INT_MAX */
#include <unistd.h>   /* getopt(), optarg */
#include <stdint.h>
#include <bits/getopt_core.h>


static void
print_usage_and_exit(const char *prog, int code)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s -rows <num_rows> -cols <num_cols> -l <lower_bound> -u <upper_bound> -o <output_file>\n"
        "  or\n"
        "  %s -r <num_rows> -c <num_cols> -l <lower_bound> -u <upper_bound> -o <output_file>\n"
        "\nExamples:\n"
        "  %s -rows 1000 -cols 512 -l -1.0 -u 1.0 -o matrix.bin\n"
        "  %s -r 1000 -c 512 -l -1.0 -u 1.0 -o matrix.bin\n",
        prog, prog, prog, prog);
    exit(code);
}

/* Parse helpers */
static long parse_long(const char *s, const char *flag_name)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        fprintf(stderr, "Error: invalid integer for %s: '%s'\n", flag_name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static double parse_double(const char *s, const char *flag_name)
{
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0') {
        fprintf(stderr, "Error: invalid double for %s: '%s'\n", flag_name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

/* Uniform random double in [a,b] */
static inline double rand_uniform(double a, double b)
{
    double r = (double)rand() / (double)RAND_MAX;
    return a + (b - a) * r;
}

/* Normalize argv: translate "-rows" -> "-r", "-cols" -> "-c".
 * We don't copy strings; we just swap the pointers to constant short flags.
 * Returns the possibly-updated argv pointer (same storage as input).
 */
static char **normalize_args(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-rows") == 0) {
            argv[i] = "-r";
        } else if (strcmp(argv[i], "-cols") == 0) {
            argv[i] = "-c";
        }
        /* The others (-l, -u, -o) are already short-form; nothing to do. */
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            /* Let getopt handle -h; accept --help by printing usage immediately. */
            if (argv[i][1] == '-') {
                print_usage_and_exit(argv[0], EXIT_SUCCESS);
            }
        }
    }
    return argv;
}

int main(int argc, char **argv)
{
    const char *prog = argv[0];

    if (argc == 1) {
        print_usage_and_exit(prog, EXIT_FAILURE);
    }

    /* Allow original flags by normalizing to short options before getopt */
    argv = normalize_args(argc, argv);

    long rows = -1;
    long cols = -1;
    double lower = 0.0;
    double upper = -1.0; /* upper < lower signals "not set" until parsed */
    const char *out_path = NULL;

    int opt;
    /* Options: r: rows, c: cols, l: lower, u: upper, o: output, h: help */
    while ((opt = getopt(argc, argv, "r:c:l:u:o:h")) != -1) {
        switch (opt) {
            case 'r':
                rows = parse_long(optarg, "-rows/-r");
                break;
            case 'c':
                cols = parse_long(optarg, "-cols/-c");
                break;
            case 'l':
                lower = parse_double(optarg, "-l");
                break;
            case 'u':
                upper = parse_double(optarg, "-u");
                break;
            case 'o':
                out_path = optarg;
                break;
            case 'h':
                print_usage_and_exit(prog, EXIT_SUCCESS);
                break;
            default:
                print_usage_and_exit(prog, EXIT_FAILURE);
        }
    }

    /* Validate required arguments */
    if (rows <= 0) {
        fprintf(stderr, "Error: -rows/-r must be a positive integer.\n");
        return EXIT_FAILURE;
    }
    if (cols <= 0) {
        fprintf(stderr, "Error: -cols/-c must be a positive integer.\n");
        return EXIT_FAILURE;
    }
    if (upper < lower) {
        fprintf(stderr, "Error: -u (upper) must be >= -l (lower).\n");
        return EXIT_FAILURE;
    }
    if (!out_path) {
        fprintf(stderr, "Error: -o <output_file> is required.\n");
        return EXIT_FAILURE;
    }

    /* Safe casts to size_t and overflow checks */
    const size_t n_rows = (size_t)rows;
    const size_t n_cols = (size_t)cols;
    if ((long)n_rows != rows || (long)n_cols != cols) {
        fprintf(stderr, "Error: rows/cols out of supported range on this platform.\n");
        return EXIT_FAILURE;
    }

    size_t n_elems;
#if defined(__has_builtin)
#  if __has_builtin(__builtin_mul_overflow)
    if (__builtin_mul_overflow(n_rows, n_cols, &n_elems)) {
        fprintf(stderr, "Error: rows*cols overflows size_t.\n");
        return EXIT_FAILURE;
    }
#  else
    n_elems = n_rows * n_cols;
#  endif
#else
    n_elems = n_rows * n_cols;
#endif

    size_t ptrs_bytes = n_rows * sizeof(double *);
    size_t payload_bytes = n_elems * sizeof(double);
    size_t total_bytes = ptrs_bytes + payload_bytes;

    /* Single allocation */
    void *block = malloc(total_bytes);
    if (!block) {
        fprintf(stderr, "Error: malloc(%zu) failed: %s\n", total_bytes, strerror(errno));
        return EXIT_FAILURE;
    }

    double **row_ptrs = (double **)block;
    double  *payload  = (double *)((unsigned char *)block + ptrs_bytes);

    /* Wire up row pointers */
    for (size_t r = 0; r < n_rows; ++r) {
        row_ptrs[r] = payload + r * n_cols;
    }

    /* Seed RNG: prefer /dev/urandom, fallback to time-based */
    {
        unsigned int seed = (unsigned int)time(NULL);
        FILE *urnd = fopen("/dev/urandom", "rb");
        if (urnd) {
            (void)fread(&seed, sizeof(seed), 1, urnd);
            fclose(urnd);
        }
        srand(seed);
    }

    /* Fill matrix */
    for (size_t r = 0; r < n_rows; ++r) {
        double *row = row_ptrs[r];
        for (size_t c = 0; c < n_cols; ++c) {
            row[c] = rand_uniform(lower, upper);
        }
    }

    /* Open output */
    FILE *fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: open '%s' for writing failed: %s\n", out_path, strerror(errno));
        free(block);
        return EXIT_FAILURE;
    }

    /* Header as two ints */
    if (rows > (long)INT_MAX || cols > (long)INT_MAX) {
        fprintf(stderr, "Error: rows/cols exceed INT_MAX; cannot store in 2 x int header.\n");
        fclose(fp);
        free(block);
        return EXIT_FAILURE;
    }
    int irows = (int)rows, icols = (int)cols;

    if (fwrite(&irows, sizeof(int), 1, fp) != 1 ||
        fwrite(&icols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error: writing header to '%s' failed: %s\n", out_path, strerror(errno));
        fclose(fp);
        free(block);
        return EXIT_FAILURE;
    }

    /* Payload in row-major order */
    size_t wrote = fwrite(payload, sizeof(double), n_elems, fp);
    if (wrote != n_elems) {
        fprintf(stderr, "Error: short write: expected %zu doubles, wrote %zu: %s\n",
                n_elems, wrote, strerror(errno));
        fclose(fp);
        free(block);
        return EXIT_FAILURE;
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "Warning: fclose('%s') failed: %s\n", out_path, strerror(errno));
    }

    free(block);

    fprintf(stdout,
            "Wrote matrix: %ld x %ld (range %.6g..%.6g) to '%s'\n"
            "Header: [int rows=%d][int cols=%d], followed by %zu doubles (row-major).\n",
            rows, cols, lower, upper, out_path, irows, icols, n_elems);

    return EXIT_SUCCESS;
}

