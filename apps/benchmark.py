from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import csv

import plotly.graph_objects as go

# --- your project imports ---
from molecsim.physics.calculations import full_potential, molecule_potential
from molecsim.data.contributors import make_potcall, SystemPotentials
from molecsim.data.properties import SystemProperties
from molecsim.physics.mutate import mutate_rotate, mutate_translate, propose_new
from molecsim.physics.potentials import coulomb, lennard_jones
from molecsim.physics.mc import metropolis
from molecsim.utils.models import tip3p
from molecsim.utils.place import randomized_system
from molecsim.utils.parsers import parse_args, parse_com
from molecsim.data.rng import RNG

# -------------------------------
# Internal sweep controls (no CLI)
# -------------------------------
# Loop (compute-only) sweep up to 5e4
_LOOP_MIN = 10
_LOOP_MAX = 50_000
_LOOP_POINTS = 14  # log-spaced points

# Full potential sweep up to 2e3
_FULL_MIN = 10
_FULL_MAX = 2_000
_FULL_POINTS = 12  # log-spaced points


def _add_order_ref_line(
    fig: go.Figure, N_list: list[int], y_list: list[float], order: int, name: str
) -> None:
    """Add an O(N^order) reference line anchored at the highest-N data point."""
    if not N_list or not y_list:
        return
    # Use the actual highest-N point as the anchor
    idx_max = int(np.argmax(N_list))
    N_max = float(N_list[idx_max])
    y_max = float(y_list[idx_max])

    Ns = np.array(sorted(set(N_list)), dtype=float)
    y_ref = y_max * (Ns / N_max) ** order

    fig.add_trace(
        go.Scatter(
            x=Ns.tolist(),
            y=y_ref.tolist(),
            mode="lines",
            name=name,
            line=dict(width=2, dash="dash"),
            showlegend=True,
        )
    )


def logspace_ints(nmin: int, nmax: int, k: int) -> list[int]:
    xs = np.logspace(np.log10(nmin), np.log10(nmax), k)
    vals = sorted(set(int(round(x)) for x in xs if x >= 1))
    if not vals:
        return [nmin, nmax] if nmin != nmax else [nmin]
    if vals[0] != nmin:
        vals[0] = nmin
    if vals[-1] != nmax:
        vals[-1] = nmax
    # strictly increasing
    clean = [vals[0]]
    for v in vals[1:]:
        if v > clean[-1]:
            clean.append(v)
    return clean


def scale_box(base_box, base_mols: int, target_mols: int) -> np.ndarray:
    base_box = np.asarray(base_box, dtype=float)
    factor = (target_mols / float(max(base_mols, 1))) ** (1.0 / 3.0)
    return base_box * factor


def make_forces(props: SystemProperties, model) -> SystemPotentials:
    return SystemPotentials(
        [
            make_potcall(coulomb, props, model),
            make_potcall(lennard_jones, props, model),
        ]
    )


def time_one_step(sim, forces, rng: RNG) -> int:
    """Compute-only single MC 'step' timing in ns."""
    mutation_functions = [mutate_rotate, mutate_translate]
    function_coefficients = [0.5, 1.0]
    t0 = time.perf_counter_ns()
    old_mol, new_mol, _ = propose_new(
        sim, mutation_functions, function_coefficients, rng
    )
    old_E = molecule_potential(sim, old_mol, forces)
    new_E = molecule_potential(sim, new_mol, forces)
    dE = new_E - old_E
    _ = metropolis(sim.properties, sim.model, dE, rng)
    t1 = time.perf_counter_ns()
    return t1 - t0


def time_full(sim, forces) -> int:
    t0 = time.perf_counter_ns()
    _ = full_potential(sim, forces)
    t1 = time.perf_counter_ns()
    return t1 - t0


def _apply_report_style(fig: go.Figure, title: str) -> None:
    """Light theme & report-friendly layout."""
    fig.update_layout(
        template="plotly_white",
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DejaVu Sans, Arial, Helvetica, sans-serif", size=14),
        legend=dict(orientation="h", x=0.0, y=1.08),
        margin=dict(l=60, r=20, t=70, b=55),
        xaxis=dict(
            type="log",
            title="Molecules (N)",
            showgrid=True,
            gridcolor="#e6e6e6",
            zeroline=False,
        ),
        yaxis=dict(
            type="log",
            title="Time (ms)",
            showgrid=True,
            gridcolor="#e6e6e6",
            zeroline=False,
        ),
    )


def _write_plot(
    fig: go.Figure,
    html_path: Path,
    png_path: Path,
    svg_path: Path,
    width_px: int = 1100,
    height_px: int = 680,
) -> None:
    """Write HTML + PNG + SVG; sizes chosen to fit reports (single column)."""
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        # For crisp print: export at explicit size; SVG is vector (best for LaTeX)
        # Requires: pip install -U kaleido
        fig.write_image(
            png_path, format="png", width=width_px, height=height_px, scale=1
        )
        fig.write_image(svg_path, format="svg", width=width_px, height=height_px)
        print(f"Wrote plots: {html_path}, {png_path}, {svg_path}")
    except Exception as e:
        print(f"Wrote HTML plot: {html_path}")
        print(
            f"(Static PNG/SVG export skipped; install 'kaleido' to enable) Reason: {e}"
        )


def main():
    # Use your parser exactly
    args = parse_args()

    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    # Outputs (separate files for loop vs full)
    loop_csv = outdir / "bench_loop_step.csv"
    full_csv = outdir / "bench_full_potential.csv"

    loop_html = outdir / "bench_loop_step.html"
    loop_png = outdir / "bench_loop_step.png"
    loop_svg = outdir / "bench_loop_step.svg"

    full_html = outdir / "bench_full_potential.html"
    full_png = outdir / "bench_full_potential.png"
    full_svg = outdir / "bench_full_potential.svg"

    model = tip3p()
    # Use ONLY the first file as prototype species
    proto = parse_com(args.files[0], model)

    base_mols = max(1, sum(args.numbers))
    base_box = np.asarray(args.box, dtype=float)

    # ---------------------------
    # Loop sweep (up to 5e4)
    # ---------------------------
    rng = RNG(args.seed)
    n_loop = logspace_ints(_LOOP_MIN, _LOOP_MAX, _LOOP_POINTS)

    loop_rows = []
    for N in n_loop:
        mols = {proto: N}
        boxN = scale_box(base_box, base_mols, N)
        props = SystemProperties(
            molecules=mols,
            box=boxN,
            temperature=args.temperature,
            periodic=args.periodic,
        )
        sim = randomized_system(props, model, rng)
        forces = make_forces(props, model)

        # Warm-up
        _ = time_one_step(sim, forces, rng)

        # Average over args.steps
        times = np.empty(args.steps, dtype=np.int64)
        for i in range(args.steps):
            times[i] = time_one_step(sim, forces, rng)
        step_ms = times.mean() * 1e-6

        loop_rows.append(
            dict(molecules=N, Lx=boxN[0], Ly=boxN[1], Lz=boxN[2], step_ms=step_ms)
        )
        print(f"[LOOP N={N:6d}] step ≈ {step_ms:9.3f} ms")

    # Write loop CSV
    with open(loop_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loop_rows[0].keys()))
        w.writeheader()
        w.writerows(loop_rows)

    # Plot loop (separate figure, light theme)
    fig_loop = go.Figure()
    # Existing main series
    fig_loop.add_trace(
        go.Scatter(
            x=[r["molecules"] for r in loop_rows],
            y=[r["step_ms"] for r in loop_rows],
            mode="lines+markers",
            name="Main-loop step (ms)",
            line=dict(width=2),
            marker=dict(size=6),
        )
    )
    # NEW: O(N) reference anchored at highest-N point
    _loop_Ns = [r["molecules"] for r in loop_rows]
    _loop_Ys = [r["step_ms"] for r in loop_rows]
    _add_order_ref_line(fig_loop, _loop_Ns, _loop_Ys, order=1, name="O(N) ref")
    _apply_report_style(fig_loop, "Benchmark: MC main-loop step")
    _write_plot(fig_loop, loop_html, loop_png, loop_svg)

    # ---------------------------
    # Full potential sweep (up to 2e3)
    # ---------------------------
    rng = RNG(args.seed + 1)  # separate RNG stream
    n_full = logspace_ints(_FULL_MIN, _FULL_MAX, _FULL_POINTS)

    full_rows = []
    for N in n_full:
        mols = {proto: N}
        boxN = scale_box(base_box, base_mols, N)
        forces = make_forces(props, model)
        props = SystemProperties(
            molecules=mols,
            box=boxN,
            temperature=args.temperature,
            periodic=args.periodic,
        )
        sim = randomized_system(props, model, forces, rng)

        # Warm-up
        _ = time_full(sim, forces)
        full_ms = time_full(sim, forces) * 1e-6

        full_rows.append(
            dict(molecules=N, Lx=boxN[0], Ly=boxN[1], Lz=boxN[2], full_ms=full_ms)
        )
        print(f"[FULL N={N:6d}] full ≈ {full_ms:9.3f} ms")

    # Write full CSV
    with open(full_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        w.writeheader()
        w.writerows(full_rows)

    # Plot full (separate figure, light theme)
    fig_full = go.Figure()
    fig_full.add_trace(
        go.Scatter(
            x=[r["molecules"] for r in full_rows],
            y=[r["full_ms"] for r in full_rows],
            mode="lines+markers",
            name="Full potential (ms)",
            line=dict(width=2),
            marker=dict(size=6),
        )
    )
    # NEW: O(N²) reference anchored at highest-N point
    _full_Ns = [r["molecules"] for r in full_rows]
    _full_Ys = [r["full_ms"] for r in full_rows]
    _add_order_ref_line(fig_full, _full_Ns, _full_Ys, order=2, name="O(N²) ref")
    _apply_report_style(fig_full, "Benchmark: full potential")
    _write_plot(fig_full, full_html, full_png, full_svg)

    print(f"\nLoop CSV: {loop_csv}")
    print(f"Full  CSV: {full_csv}")


if __name__ == "__main__":
    main()
