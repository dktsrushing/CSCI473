#!/usr/bin/env python3
"""
test_matrix_vector_mult.py

Usage:
  python3 test_matrix_vector_mult.py starting_n ending_n n_increment

What it does:
  - For each n in [start..end] step increment:
      * builds A: n x n, B: n x 1 with ./make-matrix (once per n)
      * runs ./matrix-vector-multiply A B C repeatedly
        until (trials >= 5) AND (cumulative total_time >= 2.0s)
      * parses the one-line "TIMING ..." report for each trial
      * writes per-trial rows to ./results/trial_results.csv
      * writes per-n summary (mean/stdev) to ./results/summary_results.csv
  - Plots:
      * overall_time_vs_n.png, read_time_vs_n.png, compute_time_vs_n.png, write_time_vs_n.png
      * combined_linear.png  (all four curves; linear Y; total has shaded ±1σ band)
      * combined_log.png     (all four curves; log Y; total has shaded ±1σ band)
  - Shows a live status bar with ETA that adapts using total ≈ a·n² + b.
"""

import sys
import os
import subprocess
import tempfile
import re
import csv
import time
from pathlib import Path
from math import ceil, isfinite
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

TIMING_RE = re.compile(
    r"TIMING\s+total_s=(?P<total>\d+\.\d+)\s+read_s=(?P<read>\d+\.\d+)\s+compute_s=(?P<compute>\d+\.\d+)\s+write_s=(?P<write>\d+\.\d+)\s+m=(?P<m>\d+)\s+n=(?P<n>\d+)"
)

MAKE_MATRIX = "./make-matrix"
MATVEC = "./matrix-vector-multiply"

MIN_TRIALS = 5
TARGET_CUM_SEC = 2.0  # run trials until cumulative total >= 2s (and MIN_TRIALS met)

def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

def check_tools():
    for tool in (MAKE_MATRIX, MATVEC):
        if not os.path.isfile(tool) or not os.access(tool, os.X_OK):
            die(f"Required executable not found or not executable: {tool}")

def run_cmd(cmd, cwd=None):
    result = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result.stdout, result.stderr, result.returncode

def parse_timing(s):
    for line in s.splitlines():
        m = TIMING_RE.search(line)
        if m:
            d = m.groupdict()
            return {
                "total": float(d["total"]),
                "read": float(d["read"]),
                "compute": float(d["compute"]),
                "write": float(d["write"]),
                "m": int(d["m"]),
                "n_inside": int(d["n"]),
            }
    return None

def ensure_results_dir():
    p = Path("./results")
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_matrix(path, rows, cols, lower=-1.0, upper=1.0):
    cmd = [MAKE_MATRIX, "-rows", str(rows), "-cols", str(cols),
           "-l", str(lower), "-u", str(upper), "-o", str(path)]
    out, err, rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"make-matrix failed (rc={rc})\nSTDOUT:\n{out}\nSTDERR:\n{err}")

def matvec(A_path, B_path, C_path):
    cmd = [MATVEC, str(A_path), str(B_path), str(C_path)]
    out, err, rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"matrix-vector-multiply failed (rc={rc})\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    t = parse_timing(out) or parse_timing(err)
    if t is None:
        raise RuntimeError(f"Could not parse TIMING line.\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return t

# ---------- ETA model: total_s ~ a*(n^2) + b ----------
def fit_time_n2(ns, totals):
    """Return (a,b) minimizing least-squares for y ≈ a*x + b with x=n^2."""
    if len(ns) == 0:
        return 0.0, 0.0
    if len(ns) == 1:
        x = ns[0] * ns[0]
        a = totals[0] / x if x > 0 else totals[0]
        b = 0.0
        return a, b
    xs = np.array([n*n for n in ns], dtype=float)
    ys = np.array(totals, dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    # least-squares
    sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a, b = sol
    return float(a), float(b)

def predict_time_for_n(a, b, n):
    y = a * (n*n) + b
    if not isfinite(y) or y < 0:
        y = 0.0
    return y

def predict_trials_needed(per_trial_s):
    """Trials required to satisfy both constraints."""
    if per_trial_s <= 0:
        return MIN_TRIALS
    return max(MIN_TRIALS, int(ceil(TARGET_CUM_SEC / per_trial_s)))

def render_progress(done_items, total_items, n, trial_idx, trials_needed, last_total_s, eta_s, bar_width=36):
    frac = (done_items / total_items) if total_items else 1.0
    filled = int(bar_width * frac + 0.5)
    bar = "█" * filled + "·" * (bar_width - filled)
    def fmt_sec(s):
        if s >= 3600:
            return f"{int(s//3600)}h {int((s%3600)//60)}m {int(s%60)}s"
        if s >= 60:
            return f"{int(s//60)}m {int(s%60)}s"
        return f"{s:.1f}s"
    line = (f"[{bar}] {done_items}/{total_items}  "
            f"n={n:<6}  trial={trial_idx}/{trials_needed}  "
            f"last={last_total_s:>7.3f}s  ETA={fmt_sec(eta_s):>8}")
    print("\r" + line, end="", flush=True)

def plot_series(xs, ys, title, ylabel, filename):
    plt.figure()
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Matrix size n")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_combined(ns, total_mean, read_mean, compute_mean, write_mean,
                  total_std, filename, ylog=False):
    plt.figure()
    # Lines
    plt.plot(ns, total_mean, marker="o", linewidth=2, label="Total")
    plt.plot(ns, read_mean, marker="o", linewidth=2, label="Read")
    plt.plot(ns, compute_mean, marker="o", linewidth=2, label="Multiply")
    plt.plot(ns, write_mean, marker="o", linewidth=2, label="Write")
    # Shaded variance band for total (±1σ)
    total_mean = np.array(total_mean)
    total_std = np.array(total_std)
    lower = total_mean - total_std
    upper = total_mean + total_std
    plt.fill_between(ns, lower, upper, alpha=0.2, label="Total ±1σ")
    # Styling
    plt.title("Timing vs matrix size (n)" + (" [log Y]" if ylog else ""))
    plt.xlabel("Matrix size n")
    plt.ylabel("Time (seconds)")
    if ylog:
        plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)

    try:
        n_start = int(sys.argv[1])
        n_end = int(sys.argv[2])
        n_step = int(sys.argv[3])
    except ValueError:
        die("starting_n, ending_n, n_increment must be integers")
    if n_start <= 0 or n_end <= 0 or n_step <= 0:
        die("All parameters must be positive integers.")
    if n_start > n_end:
        die("starting_n must be <= ending_n")

    check_tools()
    results_dir = ensure_results_dir()
    trial_csv_path = results_dir / "trial_results.csv"
    summary_csv_path = results_dir / "summary_results.csv"

    ns_all = list(range(n_start, n_end + 1, n_step))

    # For ETA:
    measured_ns = []          # per-trial ns (for model)
    measured_totals = []      # per-trial totals (for model)

    # CSV writers
    trial_fields = ["n", "trial_index", "total_s", "read_s", "compute_s", "write_s"]
    summary_fields = ["n", "trials", "total_mean_s", "total_std_s",
                      "read_mean_s", "compute_mean_s", "write_mean_s"]

    # Prepare containers for plots
    ns_summary = []
    total_means = []
    total_stds = []
    read_means = []
    compute_means = []
    write_means = []

    t0_wall = time.time()

    # Pre-compute an initial total_items estimate for the progress bar:
    # Use a crude guess assuming predicted trials per n with a,b=0, so need MIN_TRIALS (worst-case).
    # We'll dynamically update the fraction by counting completed (n,trial) pairs.
    total_items_est = len(ns_all) * MIN_TRIALS
    done_items = 0

    print(f"Running sweep: n from {n_start} to {n_end} by {n_step}")

    with open(trial_csv_path, "w", newline="") as f_trial, \
         open(summary_csv_path, "w", newline="") as f_summary, \
         tempfile.TemporaryDirectory() as tmpdir:

        trial_writer = csv.DictWriter(f_trial, fieldnames=trial_fields)
        summary_writer = csv.DictWriter(f_summary, fieldnames=summary_fields)
        trial_writer.writeheader()
        summary_writer.writeheader()

        tmpdir = Path(tmpdir)

        for n in ns_all:
            # Build A and B once per n
            A_path = tmpdir / f"A_{n}.bin"
            B_path = tmpdir / f"B_{n}.bin"
            C_path = tmpdir / f"C_{n}.bin"

            make_matrix(A_path, n, n, -1.0, 1.0)
            make_matrix(B_path, n, 1, -1.0, 1.0)

            totals = []
            reads = []
            computes = []
            writes = []

            cumulative_total = 0.0
            trial_idx = 0

            # Estimate trials needed for status (updated every trial)
            # We'll re-fit (a,b) after each trial.
            a, b = fit_time_n2(measured_ns, measured_totals)
            per_trial_pred = predict_time_for_n(a, b, n)
            trials_needed = predict_trials_needed(per_trial_pred)

            while (trial_idx < MIN_TRIALS) or (cumulative_total < TARGET_CUM_SEC):
                trial_idx += 1
                t = matvec(A_path, B_path, C_path)

                totals.append(t["total"])
                reads.append(t["read"])
                computes.append(t["compute"])
                writes.append(t["write"])
                cumulative_total += t["total"]

                # log per-trial
                trial_writer.writerow({
                    "n": n,
                    "trial_index": trial_idx,
                    "total_s": t["total"],
                    "read_s": t["read"],
                    "compute_s": t["compute"],
                    "write_s": t["write"],
                })

                # Update ETA model with this *trial*
                measured_ns.append(n)
                measured_totals.append(t["total"])
                a, b = fit_time_n2(measured_ns, measured_totals)

                # Update predicted trials needed for this n (could change with new info)
                per_trial_pred = predict_time_for_n(a, b, n)
                trials_needed = max(trial_idx, predict_trials_needed(per_trial_pred))

                # Predict ETA for remaining work:
                # 1) remaining trials for current n
                remaining_trials_this_n = max(0, trials_needed - trial_idx)
                eta_this_n = remaining_trials_this_n * per_trial_pred

                # 2) all future ns
                eta_future = 0.0
                for nf in ns_all[ns_all.index(n)+1:]:
                    per_trial_nf = predict_time_for_n(a, b, nf)
                    trials_nf = predict_trials_needed(per_trial_nf)
                    eta_future += trials_nf * per_trial_nf

                eta_s = eta_this_n + eta_future

                # Update a soft estimate of total_items for progress bar scaling
                total_items_est = sum(
                    max(MIN_TRIALS, predict_trials_needed(predict_time_for_n(a, b, nf)))
                    for nf in ns_all
                )
                done_items += 1  # we've completed one (n,trial) item
                render_progress(done_items, total_items_est, n, trial_idx, trials_needed, t["total"], eta_s)

            # After trials for this n, summarize
            ns_summary.append(n)
            total_means.append(mean(totals))
            total_stds.append(pstdev(totals) if len(totals) > 1 else 0.0)
            read_means.append(mean(reads))
            compute_means.append(mean(computes))
            write_means.append(mean(writes))

            summary_writer.writerow({
                "n": n,
                "trials": trial_idx,
                "total_mean_s": total_means[-1],
                "total_std_s": total_stds[-1],
                "read_mean_s": read_means[-1],
                "compute_mean_s": compute_means[-1],
                "write_mean_s": write_means[-1],
            })

    # newline after status bar
    print()

    # Make the original single-curve plots (means)
    plot_series(ns_summary, total_means,  "Overall elapsed time vs matrix size (n)", "Time (seconds)",
                results_dir / "overall_time_vs_n.png")
    plot_series(ns_summary, read_means,   "Read time vs matrix size (n)",            "Time (seconds)",
                results_dir / "read_time_vs_n.png")
    plot_series(ns_summary, compute_means,"Multiply time vs matrix size (n)",        "Time (seconds)",
                results_dir / "compute_time_vs_n.png")
    plot_series(ns_summary, write_means,  "Write time vs matrix size (n)",           "Time (seconds)",
                results_dir / "write_time_vs_n.png")

    # Combined plots with variance shading on Total
    plot_combined(ns_summary, total_means, read_means, compute_means, write_means,
                  total_stds, results_dir / "combined_linear.png", ylog=False)
    plot_combined(ns_summary, total_means, read_means, compute_means, write_means,
                  total_stds, results_dir / "combined_log.png", ylog=True)

    elapsed = time.time() - t0_wall
    print("Sweep complete.")
    print(f"- Per-trial CSV: {trial_csv_path}")
    print(f"- Summary CSV:   {summary_csv_path}")
    print(f"- Plots saved in: {results_dir}/")
    for fn in [
        "overall_time_vs_n.png",
        "read_time_vs_n.png",
        "compute_time_vs_n.png",
        "write_time_vs_n.png",
        "combined_linear.png",
        "combined_log.png",
    ]:
        print(f"  - {fn}")
    print(f"- Wall time: {elapsed:.3f}s")

if __name__ == "__main__":
    main()

