#!/usr/bin/env python3
"""
Tiny Numba @njit benchmark: pure Python vs NumPy vs Numba.

Run:
  python benchmark_njit.py
Optional:
  python benchmark_njit.py --n 2000000 --repeats 7
"""

import argparse
import time
import numpy as np

try:
    from numba import njit
except Exception as e:
    raise SystemExit(
        "Numba is required for this benchmark.\n"
        "Install: pip install numba\n"
        f"Import error: {e}"
    )

# ----------------------------
# Workload: sum of squares
# ----------------------------


def py_sum_squares(x):
    s = 0.0
    for v in x:
        s += float(v) * float(v)
    return s


def np_sum_squares(x):
    # uses vectorized multiply + sum (typically hits SIMD/BLAS-ish paths)
    return float(np.sum(x * x))


@njit(cache=True, fastmath=True, nogil=True)
def nb_sum_squares(x):
    s = 0.0
    for i in range(x.shape[0]):
        v = x[i]
        s += v * v
    return s


# ----------------------------
# Timing helpers
# ----------------------------


def time_fn(fn, *args, repeats=5):
    # run once outside to reduce cache effects
    fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def fmt_stats(times):
    # robust-ish: median + min
    med = float(np.median(times))
    tmin = float(np.min(times))
    return med, tmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000, help="array length")
    ap.add_argument("--repeats", type=int, default=7, help="timing repeats")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    x = rng.random(args.n, dtype=np.float64)

    # Warmup compile for Numba (compile cost is NOT included in timings below)
    nb_sum_squares(x)

    py_t = time_fn(py_sum_squares, x, repeats=args.repeats)
    np_t = time_fn(np_sum_squares, x, repeats=args.repeats)
    nb_t = time_fn(nb_sum_squares, x, repeats=args.repeats)

    py_med, py_min = fmt_stats(py_t)
    np_med, np_min = fmt_stats(np_t)
    nb_med, nb_min = fmt_stats(nb_t)

    # Verify correctness (within floating tolerance)
    py_val = py_sum_squares(x)
    np_val = np_sum_squares(x)
    nb_val = nb_sum_squares(x)
    if not (
        abs(py_val - np_val) / (abs(np_val) + 1e-12) < 1e-12
        and abs(nb_val - np_val) / (abs(np_val) + 1e-12) < 1e-12
    ):
        raise SystemExit("Mismatch between implementations!")

    print(f"n={args.n:,} repeats={args.repeats}")
    print("")
    print("Median time (s)   [min time in parentheses]")
    print(f"  pure python : {py_med:.6f}  ({py_min:.6f})")
    print(f"  numpy       : {np_med:.6f}  ({np_min:.6f})")
    print(f"  numba njit  : {nb_med:.6f}  ({nb_min:.6f})")
    print("")
    print("Speedup vs pure python (median):")
    print(f"  numpy      : {py_med / np_med:8.2f}x")
    print(f"  numba njit : {py_med / nb_med:8.2f}x")
    print("")
    print("Speedup numba vs numpy (median):")
    print(f"  numba/numpy: {np_med / nb_med:8.2f}x")


if __name__ == "__main__":
    main()
