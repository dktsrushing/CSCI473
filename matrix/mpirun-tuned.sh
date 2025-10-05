#!/bin/bash
mpirun \
  --mca coll_tuned_use_dynamic_rules 1 \
  --mca coll_tuned_bcast_algorithm 2 \
  --mca coll_tuned_scatterv_algorithm 1 \
  --mca coll_tuned_gatherv_algorithm 1 \
  --mca mpi_leave_pinned 1 \
  "$@"
