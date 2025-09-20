# =========================
# Parameterized Makefile
# =========================

# --- Toolchains (override on command line if desired) ---
CC      ?= gcc
MPICC   ?= mpicc

# --- Common flags ---
CSTD        ?= -std=c99
WARNFLAGS   ?= -Wall -Wextra
OPTFLAGS    ?=
CFLAGS      ?= $(CSTD) $(WARNFLAGS) $(OPTFLAGS)
LDFLAGS     ?=
LDLIBS      ?=
# Note: If your platform needs -lrt for clock_gettime, do:
#   make LDLIBS_EXTRA="-lrt"
LDLIBS_EXTRA ?=

# --- Where to place binaries ---
BIN_DIR ?= .

# --- Sources ---
SRCS_SERIAL := \
    make-matrix.c \
    print-matrix.c \
    matrix-vector-multiply.c

SRCS_MPI := \
    mpi-matrix-vector-multiply.c

# --- Targets ---
PROGS_SERIAL := $(addprefix $(BIN_DIR)/, \
    make-matrix \
    print-matrix \
    matrix-vector-multiply)

PROGS_MPI := $(addprefix $(BIN_DIR)/, \
    mpi-matrix-vector-multiply)

PROGS := $(PROGS_SERIAL) $(PROGS_MPI)

# Default target
.PHONY: all
all: $(PROGS)

# -----------------------------
# Build rules: serial programs
# -----------------------------
$(BIN_DIR)/make-matrix: make-matrix.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS) $(LDLIBS_EXTRA)

$(BIN_DIR)/print-matrix: print-matrix.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS) $(LDLIBS_EXTRA)

$(BIN_DIR)/matrix-vector-multiply: matrix-vector-multiply.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS) $(LDLIBS_EXTRA)

# -----------------------------
# Build rules: MPI program
# -----------------------------
# Allow separate flags for MPI if desired
MPICFLAGS   ?= $(CFLAGS)
MPILDFLAGS  ?= $(LDFLAGS)
MPILDLIBS   ?= $(LDLIBS)

$(BIN_DIR)/mpi-matrix-vector-multiply: mpi-matrix-vector-multiply.c
	$(MPICC) $(MPICFLAGS) $< -o $@ $(MPILDFLAGS) $(MPILDLIBS) $(LDLIBS_EXTRA)

# -----------------------------
# Convenience targets
# -----------------------------
.PHONY: debug release
debug:
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CSTD) $(WARNFLAGS) -O0 -g3 -fsanitize=address,undefined" \
	        LDFLAGS="$(LDFLAGS)" \
	        LDLIBS="$(LDLIBS)" \
	        LDLIBS_EXTRA="$(LDLIBS_EXTRA)"

release:
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CSTD) $(WARNFLAGS) -O3" \
	        LDFLAGS="$(LDFLAGS)" \
	        LDLIBS="$(LDLIBS)" \
	        LDLIBS_EXTRA="$(LDLIBS_EXTRA)"

# Clean
.PHONY: clean
clean:
	$(RM) $(PROGS) *.o *.obj *.out *.exe

