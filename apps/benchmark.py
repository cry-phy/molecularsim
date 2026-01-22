from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

# --- your project imports ---
from molecsim.data.contributors import SystemPotentials, make_potcall
from molecsim.data.properties import SystemProperties
from molecsim.data.rng import RNG
from molecsim.physics.calculations import full_potential, molecule_potential
from molecsim.physics.mc import metropolis
from molecsim.physics.mutate import mutate_rotate, mutate_translate, propose_new
from molecsim.physics.potentials import (
    coulomb,
    coulomb_dense,
    coulomb_gpu,
    coulomb_njit,
    coulomb_stream,
    lennard_jones,
    lennard_jones_dense,
    lennard_jones_gpu,
    lennard_jones_njit,
    lennard_jones_stream,
)
from molecsim.utils.models import tip3p
from molecsim.utils.parsers import parse_args, parse_com
from molecsim.utils.place import randomized_system

# -------------------------------
# Internal sweep controls (no CLI)
# -------------------------------
_LOOP_MIN = 100
_LOOP_MAX = 3_000_000
_LOOP_POINTS = 15  # log-spaced points

_FULL_MIN = 10
_FULL_MAX = 10_000
_FULL_POINTS = 12  # log-spaced points

DARK_MODE = True

# Plot colors
COLORS = pc.qualitative.Dark24


def _add_order_ref_line(
    fig: go.Figure,
    N_list: list[int],
    y_list: list[float],
    order: int,
    *,
    color: str,
    showlegend: bool = False,
    name: str = "Asymptote O(N^k)",
) -> None:
    """Add an O(N^order) reference line anchored at the highest-N point."""
    if not N_list or not y_list:
        return

    idx_max = int(np.argmax(np.asarray(N_list, dtype=float)))
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
            line=dict(width=2, dash="dash", color=color),
            showlegend=showlegend,
            hoverinfo="skip",
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
    clean = [vals[0]]
    for v in vals[1:]:
        if v > clean[-1]:
            clean.append(v)
    return clean


def scale_box(base_box, base_mols: int, target_mols: int) -> np.ndarray:
    base_box = np.asarray(base_box, dtype=float)
    factor = (target_mols / float(max(base_mols, 1))) ** (1.0 / 3.0)
    return base_box * factor


def make_simple(props: SystemProperties, model) -> SystemPotentials:
    return SystemPotentials(
        [
            make_potcall(coulomb, props, model),
            make_potcall(lennard_jones, props, model),
        ]
    )


def make_njit(props: SystemProperties, model) -> SystemPotentials:
    return SystemPotentials(
        [
            make_potcall(coulomb_njit, props, model),
            make_potcall(lennard_jones_njit, props, model),
        ]
    )


def make_dense(props: SystemProperties, model) -> SystemPotentials:
    return SystemPotentials(
        [
            make_potcall(coulomb_dense, props, model),
            make_potcall(lennard_jones_dense, props, model),
        ]
    )


def make_stream(props: SystemProperties, model) -> SystemPotentials:
    return SystemPotentials(
        [
            make_potcall(coulomb_stream, props, model),
            make_potcall(lennard_jones_stream, props, model),
        ]
    )


def make_gpu(props: SystemProperties, model) -> SystemPotentials:
    return SystemPotentials(
        [
            make_potcall(coulomb_gpu, props, model),
            make_potcall(lennard_jones_gpu, props, model),
        ]
    )


def preflight_forces(
    force_sets,
    props: SystemProperties,
    model,
    proto,
    rng: RNG,
) -> None:
    """Run each force set once to catch errors before benchmarking."""
    print("\n=== PREFLIGHT: validating force sets ===")
    for key, label, builder, *_ in force_sets:
        try:
            forces = builder(props, model)
            sim = randomized_system(props, model, forces, rng)

            old_mol, new_mol, _ = propose_new(
                sim,
                [mutate_rotate, mutate_translate],
                [0.5, 1.0],
                rng,
            )

            _ = molecule_potential(sim, old_mol, forces)
            _ = molecule_potential(sim, new_mol, forces)
            _ = full_potential(sim, forces)

            print(f"[OK] {key:>8s} — {label}")
        except Exception as e:
            print(f"[FAIL] {key:>8s} — {label}")
            raise RuntimeError(
                f"Preflight failed for force set '{key}' ({label})"
            ) from e


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
    dE = new_E.potential - old_E.potential
    _ = metropolis(sim.properties, sim.model, dE, rng)

    t1 = time.perf_counter_ns()
    return t1 - t0


def time_full(sim, forces) -> int:
    t0 = time.perf_counter_ns()
    _ = full_potential(sim, forces)
    t1 = time.perf_counter_ns()
    return t1 - t0


def _apply_report_style(fig: go.Figure, title: str) -> None:
    if DARK_MODE:
        grid = "#44475a"
        fig.update_layout(
            template="plotly_dark",
            title=title,
            paper_bgcolor="#282a36",  # Dracula-ish
            plot_bgcolor="#282a36",
            font=dict(
                family="DejaVu Sans, Arial, Helvetica, sans-serif",
                size=14,
                color="#f8f8f2",
            ),
            legend=dict(
                orientation="v",
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
            ),
            margin=dict(l=60, r=260, t=70, b=55),
        )
    else:
        grid = "#e6e6e6"
        fig.update_layout(
            template="plotly_white",
            title=title,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                family="DejaVu Sans, Arial, Helvetica, sans-serif",
                size=14,
                color="black",
            ),
            legend=dict(
                orientation="v",
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
            ),
            margin=dict(l=60, r=260, t=70, b=55),
        )

    fig.update_xaxes(
        type="log",
        title="Molecules (N)",
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
    )
    fig.update_yaxes(
        type="log",
        title="Time (ms)",
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
    )


def _set_log_ymin(fig: go.Figure, all_y: list[float], ymin_ms: float = 0.1) -> None:
    """Force y-axis to start at ymin_ms (log axis)."""
    if not all_y:
        return
    ymax = max(all_y)
    if ymax <= 0.0:
        return
    y0 = float(np.log10(ymin_ms))
    y1 = float(np.log10(ymax * 1.2))
    fig.update_yaxes(range=[y0, y1])


def _write_plot(
    fig: go.Figure,
    html_path: Path,
    png_path: Path,
    svg_path: Path,
    width_px: int = 1200,
    height_px: int = 680,
) -> None:
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
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
    import numba

    args = parse_args()

    # Let the benchmark script control numba threads.
    # (Also consider setting NUMBA_NUM_THREADS / OMP_NUM_THREADS in your shell for repeatability.)
    if hasattr(args, "threads") and args.threads is not None:
        numba.set_num_threads(int(args.threads))

    print("=== Config ===")
    print(f"Threads: {numba.get_num_threads()}")

    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    model = tip3p()
    proto = parse_com(args.files[0], model)

    base_mols = max(1, sum(args.numbers))
    base_box = np.asarray(args.box, dtype=float)

    # ---------------------------------------------------
    # Force sets:
    # Each entry: (key, label, builder, ref_order_loop, ref_order_full)
    # ---------------------------------------------------
    force_sets = [
        ("gpu", "Coulomb + LJ (gpu)", make_gpu, 1, 2),
        ("stream", "Coulomb + LJ (stream)", make_stream, 1, 2),
        ("dense", "Coulomb + LJ (dense)", make_dense, 1, 2),
        ("njit", "Coulomb + LJ (njit)", make_njit, 1, 2),
        ("py", "Coulomb + LJ (Python)", make_simple, 1, 2),
    ]

    # ---------------------------
    # PREFLIGHT CHECK
    # ---------------------------
    rng_pre = RNG(args.seed + 999)

    pre_N = 100
    mols = {proto: pre_N}
    boxN = scale_box(base_box, base_mols, pre_N)

    props_pre = SystemProperties(
        molecules=mols,
        box=boxN,
        temperature=args.temperature,
        periodic=args.periodic,
    )

    preflight_forces(force_sets, props_pre, model, proto, rng_pre)

    # ---------------------------
    # Loop sweep
    # ---------------------------
    rng = RNG(args.seed)
    n_loop = logspace_ints(_LOOP_MIN, _LOOP_MAX, _LOOP_POINTS)

    loop_csv = outdir / "bench_loop_step.csv"
    loop_rows: list[dict] = []

    print("\n=== LOOP SWEEP (main-loop step) ===")
    for N in n_loop:
        mols = {proto: N}
        boxN = scale_box(base_box, base_mols, N)
        props = SystemProperties(
            molecules=mols,
            box=boxN,
            temperature=args.temperature,
            periodic=args.periodic,
        )

        for key, label, builder, _, _ in force_sets:
            forces = builder(props, model)
            sim = randomized_system(props, model, forces, rng)

            _ = time_one_step(sim, forces, rng)  # warm-up

            times = np.empty(int(args.steps), dtype=np.int64)
            for i in range(int(args.steps)):
                times[i] = time_one_step(sim, forces, rng)
            step_ms = float(times.mean() * 1e-6)

            loop_rows.append(
                dict(
                    kind="loop",
                    force_key=key,
                    force_label=label,
                    molecules=int(N),
                    Lx=float(boxN[0]),
                    Ly=float(boxN[1]),
                    Lz=float(boxN[2]),
                    time_ms=step_ms,
                )
            )
            print(f"[{key:>8s} LOOP N={N:6d}] step ≈ {step_ms:9.3f} ms")

    with open(loop_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loop_rows[0].keys()))
        w.writeheader()
        w.writerows(loop_rows)

    # Plot loop (single plot)
    fig_loop = go.Figure()
    all_loop_y: list[float] = []

    # One dashed legend entry representing all asymptotes
    fig_loop.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Asymptote O(N^k)",
            line=dict(width=2, dash="dash", color="#aaaaaa"),
            showlegend=True,
            hoverinfo="skip",
        )
    )

    for idx, (key, label, _, ref_loop_order, _) in enumerate(force_sets):
        rows_k = [r for r in loop_rows if r["force_key"] == key]
        Ns = [int(r["molecules"]) for r in rows_k]
        Ys = [float(r["time_ms"]) for r in rows_k]
        all_loop_y.extend(Ys)

        c = COLORS[idx % len(COLORS)]

        # One legend entry per benchmark series
        fig_loop.add_trace(
            go.Scatter(
                x=Ns,
                y=Ys,
                mode="lines+markers",
                name=label,
                line=dict(width=2, color=c),
                marker=dict(size=6, color=c),
                legendgroup=key,
                showlegend=True,
            )
        )

        # Asymptote in same color, but no separate legend entry
        _add_order_ref_line(
            fig_loop,
            Ns,
            Ys,
            order=int(ref_loop_order),
            color=c,
            showlegend=False,
        )

    _apply_report_style(fig_loop, "Benchmark: MC main-loop step (multiple force sets)")
    _set_log_ymin(fig_loop, all_loop_y, ymin_ms=0.1)

    loop_html = outdir / "bench_loop_step_multi.html"
    loop_png = outdir / "bench_loop_step_multi.png"
    loop_svg = outdir / "bench_loop_step_multi.svg"
    _write_plot(fig_loop, loop_html, loop_png, loop_svg)

    # ---------------------------
    # Full potential sweep
    # ---------------------------
    rng = RNG(args.seed + 1)
    n_full = logspace_ints(_FULL_MIN, _FULL_MAX, _FULL_POINTS)

    full_csv = outdir / "bench_full_potential.csv"
    full_rows: list[dict] = []

    print("\n=== FULL SWEEP (full potential) ===")
    for N in n_full:
        mols = {proto: N}
        boxN = scale_box(base_box, base_mols, N)
        props = SystemProperties(
            molecules=mols,
            box=boxN,
            temperature=args.temperature,
            periodic=args.periodic,
        )

        for key, label, builder, _, _ in force_sets:
            forces = builder(props, model)
            sim = randomized_system(props, model, forces, rng)

            _ = time_full(sim, forces)  # warm-up
            full_ms = float(time_full(sim, forces) * 1e-6)

            full_rows.append(
                dict(
                    kind="full",
                    force_key=key,
                    force_label=label,
                    molecules=int(N),
                    Lx=float(boxN[0]),
                    Ly=float(boxN[1]),
                    Lz=float(boxN[2]),
                    time_ms=full_ms,
                )
            )
            print(f"[{key:>8s} FULL N={N:6d}] full ≈ {full_ms:9.3f} ms")

    with open(full_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        w.writeheader()
        w.writerows(full_rows)

    # Plot full (single plot)
    fig_full = go.Figure()
    all_full_y: list[float] = []

    # One dashed legend entry representing all asymptotes
    fig_full.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Asymptote O(N^k)",
            line=dict(width=2, dash="dash", color="#aaaaaa"),
            showlegend=True,
            hoverinfo="skip",
        )
    )

    for idx, (key, label, _, _, ref_full_order) in enumerate(force_sets):
        rows_k = [r for r in full_rows if r["force_key"] == key]
        Ns = [int(r["molecules"]) for r in rows_k]
        Ys = [float(r["time_ms"]) for r in rows_k]
        all_full_y.extend(Ys)

        c = COLORS[idx % len(COLORS)]

        fig_full.add_trace(
            go.Scatter(
                x=Ns,
                y=Ys,
                mode="lines+markers",
                name=label,
                line=dict(width=2, color=c),
                marker=dict(size=6, color=c),
                legendgroup=key,
                showlegend=True,
            )
        )

        _add_order_ref_line(
            fig_full,
            Ns,
            Ys,
            order=int(ref_full_order),
            color=c,
            showlegend=False,
        )

    _apply_report_style(fig_full, "Benchmark: full potential (multiple force sets)")
    _set_log_ymin(fig_full, all_full_y, ymin_ms=0.1)

    full_html = outdir / "bench_full_potential_multi.html"
    full_png = outdir / "bench_full_potential_multi.png"
    full_svg = outdir / "bench_full_potential_multi.svg"
    _write_plot(fig_full, full_html, full_png, full_svg)

    print(f"\nLoop CSV: {loop_csv}")
    print(f"Full  CSV: {full_csv}")


if __name__ == "__main__":
    main()
