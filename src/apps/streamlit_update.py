"""Streamlit app for live tracking a molecular MC system.

Install (inside your project):
    pip install .[apps]

Run:
    streamlit run src/apps/streamlit_update.py
"""

from __future__ import annotations

import csv
import io
import pickle
import threading
import time
from datetime import datetime
from typing import Any, Callable, Sequence, Tuple
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ---- Project imports ----
from molecsim.data.contributors import make_potcall, SystemPotentials
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.calculations import (
    all_rdfs,
    molecule_potential,
)
from molecsim.physics.mc import metropolis
from molecsim.physics.mutate import mutate_rotate, mutate_translate, propose_new
from molecsim.physics.potentials import coulomb, lennard_jones
from molecsim.utils.models import tip3p
from molecsim.utils.parsers import parse_com
from molecsim.utils.place import randomized_system
from molecsim.utils.plot import plot_system
from molecsim.data.rng import RNG


# --------------------------------------------------------------------------------------
# Core: system construction & single step
# --------------------------------------------------------------------------------------
def build_system(
    files: Sequence[str],
    numbers: Sequence[int],
    box: Sequence[float],
    temperature: float,
    seed: int = 10,
    periodic: bool = True,
) -> Tuple[
    Any,
    SystemProperties,
    SystemPotentials,
    RNG,
    go.Figure,
    Callable[[Any], go.Figure | None],
    ModelConstants,
]:  # ← add model in return
    """Build a randomized simulation system and initial figure from inputs."""
    # NEW: choose a model/constants (e.g., TIP3P)
    model = tip3p()

    # parse_com now needs the model (registry/charges)
    molecule_list = {
        parse_com(path, model): n for path, n in zip(files, numbers, strict=True)
    }

    rng = RNG(int(seed))
    box_arr = np.asarray(box, dtype=float)
    temp = float(temperature)

    # NEW: SystemProperties(box, T, periodic, constants=model)
    props = SystemProperties(molecule_list, box_arr, temp, periodic)

    # NEW: randomized_system(props, model, rng)
    sim = randomized_system(props, model, rng)

    # NEW: make_potcall(fn, props, model)
    forces = SystemPotentials(
        [
            make_potcall(coulomb, props, model),
            make_potcall(lennard_jones, props, model),
        ]
    )

    fig, update = plot_system(sim)
    return sim, props, forces, rng, fig, update, model


def step_once(
    sim: Any,
    forces: SystemPotentials,
    rng: RNG,
    mutation_functions: Sequence[Callable[..., Any]],
    function_coefficients: np.ndarray,
    ema_accept: np.ndarray,
    alpha: float,
    stats: dict[str, int],
) -> None:
    """Perform one Metropolis MC proposal and update stats/EMA."""
    old_molecule, new_molecule, index = propose_new(
        sim, mutation_functions, function_coefficients, rng
    )

    old_contrib = molecule_potential(sim, old_molecule, forces)
    new_contrib = molecule_potential(sim, new_molecule, forces)
    diff = new_contrib - old_contrib

    decider = metropolis(sim.properties, sim.model, diff, rng)

    ema = ema_accept[index]
    ema = ema + alpha * (int(decider) - ema)
    ema_accept[index] = float(np.clip(ema, 0.0, 1.0))

    if decider:
        old_molecule.copy_from(new_molecule)
        stats["accepted"] += 1
    stats["trials"] += 1


# --------------------------------------------------------------------------------------
# State initialization & helpers
# --------------------------------------------------------------------------------------
def configure_page() -> None:
    """Configure Streamlit page and title."""
    st.set_page_config(page_title="Molecular MC", layout="wide")
    st.title("Molecular MC — Live Viewer")


def ensure_state() -> None:
    """Initialize Streamlit session_state with defaults once."""
    if "initialized" in st.session_state:
        return

    # Sim & physics
    st.session_state.initialized = False
    st.session_state.sim = None
    st.session_state.props = None
    st.session_state.model = None
    st.session_state.forces = None
    st.session_state.rng = None

    # Mutations & acceptance control
    st.session_state.mutation_functions = [mutate_rotate, mutate_translate]
    st.session_state.function_coefficients = np.array([0.5, 1.0], dtype=float)
    st.session_state.recent_acceptance = np.ones(2, dtype=float)
    st.session_state.alpha = 1e-3
    st.session_state.stats = {"accepted": 0, "trials": 0}

    # Plot persistence
    st.session_state.fig: go.Figure | None = None
    st.session_state.update_fig: Callable[[Any], go.Figure | None] | None = None
    st.session_state.scene_camera: dict | None = None
    st.session_state.apply_camera_next: bool = False

    # UI cadence
    st.session_state.redraw_interval_s = 5.0  # seconds

    # Worker thread
    st.session_state.worker_running = False
    st.session_state.worker_stop_evt = threading.Event()
    st.session_state.worker_lock = threading.Lock()
    st.session_state.worker_thread: threading.Thread | None = None
    st.session_state.worker_batch = 500

    # Manual stepping
    st.session_state.manual_steps = 200

    # Save/load defaults
    st.session_state.setdefault(
        "save_filename", f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pickle"
    )

    # RDF recorder state & config
    st.session_state.rdf_recording = False
    st.session_state.rdf_bins_count = 200
    st.session_state.rdf_bins_edges = None
    st.session_state.rdf_accum = {}
    st.session_state.rdf_samples = 0
    st.session_state.rdf_stride = 100
    st.session_state.rdf_auto_sample = True
    st.session_state.rdf_last_sample_step = 0
    st.session_state.setdefault(
        "rdf_csv_filename", f"rdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    st.session_state.setdefault("rdf_csv_normalize", True)

    # Coefficient editor default
    st.session_state.setdefault("coef_reset_ema_on_apply", False)

    # Help panel
    st.session_state.setdefault("show_quickstart", True)


# --------------------------------------------------------------------------------------
# Worker lifecycle
# --------------------------------------------------------------------------------------
def worker_loop() -> None:
    """Background loop that advances the simulation in batches."""
    while True:
        try:
            stop_evt = st.session_state.worker_stop_evt
            lock = st.session_state.worker_lock
            if stop_evt.is_set():
                break
            initialized = st.session_state.get("initialized", False)
        except Exception:  # session_state not yet ready during a rerun window
            time.sleep(0.02)
            continue

        if not initialized:
            time.sleep(0.01)
            continue

        try:
            with lock:
                sim = st.session_state.sim
                forces = st.session_state.forces
                rng = st.session_state.rng
                muts = st.session_state.mutation_functions
                coeffs = st.session_state.function_coefficients
                ema = st.session_state.recent_acceptance
                alpha = float(st.session_state.alpha)
                stats = st.session_state.stats

                if sim is None or forces is None or rng is None:
                    time.sleep(0.01)
                    continue

                for _ in range(int(st.session_state.worker_batch)):
                    step_once(sim, forces, rng, muts, coeffs, ema, alpha, stats)

                    # Auto-sample RDF while recording
                    if (
                        st.session_state.rdf_recording
                        and st.session_state.rdf_auto_sample
                    ):
                        stride = max(1, int(st.session_state.rdf_stride))
                        now_steps = int(st.session_state.stats.get("trials", 0))
                        last = int(st.session_state.rdf_last_sample_step)
                        if now_steps - last >= stride:
                            rdf_record_single_snapshot(
                                int(st.session_state.rdf_bins_count)
                            )
                            st.session_state.rdf_last_sample_step = now_steps
        except Exception:
            time.sleep(0.02)
            continue

        time.sleep(0.001)


def start_worker() -> None:
    """Start the background worker thread if not running."""
    if st.session_state.worker_running:
        return
    st.session_state.worker_stop_evt.clear()
    th = threading.Thread(target=worker_loop, daemon=True)
    if add_script_run_ctx is not None:
        try:
            add_script_run_ctx(th)
        except Exception:
            pass
    th.start()
    st.session_state.worker_thread = th
    st.session_state.worker_running = True


def stop_worker() -> None:
    """Stop the background worker thread if running."""
    if not st.session_state.worker_running:
        return
    st.session_state.worker_stop_evt.set()
    th = st.session_state.worker_thread
    if th is not None and th.is_alive():
        th.join(timeout=1.0)
    st.session_state.worker_thread = None
    st.session_state.worker_running = False


# --------------------------------------------------------------------------------------
# Serialization & RDF utilities
# --------------------------------------------------------------------------------------
def pickle_sim_bytes() -> bytes:
    """Serialize the current simulation to bytes under a lock."""
    with st.session_state.worker_lock:
        bio = io.BytesIO()
        pickle.dump(st.session_state.sim, bio, protocol=pickle.HIGHEST_PROTOCOL)
        return bio.getvalue()


def save_sim_to_path(path: str) -> None:
    """Serialize the current simulation to a file path."""
    with st.session_state.worker_lock:
        with open(path, "wb") as f:
            pickle.dump(st.session_state.sim, f, protocol=pickle.HIGHEST_PROTOCOL)


import numpy as np
import plotly.graph_objects as go
import streamlit as st


def rdf_record_single_snapshot(num_bins: int) -> None:
    """Accumulate one RDF snapshot into session accumulators (raw counts)."""
    bins, snap = all_rdfs(
        st.session_state.sim, num_bins
    )  # bins may be edges *or* centers

    # (Re)initialize accumulators when binning changes
    if st.session_state.rdf_bins_edges is None or len(
        st.session_state.rdf_bins_edges
    ) != len(bins):
        st.session_state.rdf_bins_edges = np.asarray(bins, dtype=float)
        st.session_state.rdf_accum = {}
        st.session_state.rdf_samples = 0

    for key, h in snap.items():
        h = np.asarray(h, dtype=float)
        if (
            key not in st.session_state.rdf_accum
            or st.session_state.rdf_accum[key].shape != h.shape
        ):
            st.session_state.rdf_accum[key] = np.zeros_like(h, dtype=float)
        st.session_state.rdf_accum[key] += h

    st.session_state.rdf_samples += 1


def _bin_geometry(bins: np.ndarray, counts_len: int):
    """Return (edges, centers, dr, shell_vol) consistent with counts length.

    Detects if `bins` are centers (len == counts_len) or edges (len == counts_len+1).
    Assumes ~uniform spacing if centers are given.
    """
    bins = np.asarray(bins, dtype=float)

    if len(bins) == counts_len + 1:
        # bins are edges
        edges = bins
        centers = 0.5 * (edges[:-1] + edges[1:])
        dr = np.diff(edges)
    elif len(bins) == counts_len:
        # bins are centers -> infer edges assuming uniform spacing
        centers = bins
        # robust spacing: median to tolerate small FP noise
        dr_scalar = float(np.median(np.diff(centers))) if len(centers) > 1 else 1.0
        edges = np.empty(counts_len + 1, dtype=float)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - 0.5 * dr_scalar
        edges[-1] = centers[-1] + 0.5 * dr_scalar
        dr = np.diff(edges)
    else:
        raise ValueError(
            f"Incompatible bins/counts: len(bins)={len(bins)}, counts_len={counts_len}."
            " Expected len(bins)==counts_len or counts_len+1."
        )

    shell_vol = 4.0 * np.pi * centers**2 * dr  # per-bin spherical shell volumes
    return edges, centers, dr, shell_vol


def rdf_plot() -> go.Figure | None:
    """Return a Plotly figure of averaged, normalized RDF g_AB(r)."""
    with st.session_state.worker_lock:
        bins = st.session_state.rdf_bins_edges
        samples = int(st.session_state.rdf_samples)
        accum = {k: v.copy() for k, v in st.session_state.rdf_accum.items()}

    if bins is None or samples == 0 or not accum:
        return None

    # System numbers for normalization
    sim = st.session_state.sim
    V = float(np.prod(sim.properties.box))
    atoms = sim.data.atoms
    to_id = sim.data.atom_str_to_ids

    def normalize_counts(pair, counts, shell_vol):
        a, b = pair
        ida = to_id([a])[0]
        idb = to_id([b])[0]
        NA = int(np.count_nonzero(atoms == ida))
        NB = int(np.count_nonzero(atoms == idb))
        if NA == 0 or NB == 0:
            return np.zeros_like(shell_vol)

        if ida == idb:
            rhoB = (NA - 1) / V  # exclude self for A==A
            norm = NA * rhoB
        else:
            rhoB = NB / V
            norm = NA * rhoB

        mean_counts = counts / max(samples, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = mean_counts / (norm * shell_vol)
            g[~np.isfinite(g)] = 0.0
        return g

    # Build curves and compute a robust y-cap
    curves = []
    all_vals = []

    for (a, b), counts in sorted(accum.items()):
        counts = np.asarray(counts, dtype=float)
        _, centers, _, shell_vol = _bin_geometry(bins, counts_len=len(counts))
        g = normalize_counts((a, b), counts, shell_vol)
        curves.append((f"{a}-{b}", centers, g))
        all_vals.append(g[np.isfinite(g)])

    if all_vals:
        all_vals = np.concatenate(all_vals)
        y_cap = float(np.nanpercentile(all_vals, 95)) * 1.15
        if not np.isfinite(y_cap) or y_cap <= 0:
            y_cap = 3.0
    else:
        y_cap = 3.0

    # x-range: PBC-safe bulk view
    r_max = 0.5 * float(np.min(sim.properties.box)) if sim.properties.periodic else None

    fig = go.Figure()
    for name, x, y in curves:
        # if r_max set, mask to that range
        if r_max is not None:
            m = (x >= 0.0) & (x <= r_max)
            x, y = x[m], y[m]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))

    layout = dict(
        title=f"RDF (normalized g(r); bins={len(curves[0][1]) if curves else 0}, samples={samples})",
        xaxis_title="r (Å)",
        yaxis_title="g(r)",
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0.0, y_cap]),
    )
    if r_max is not None:
        layout["xaxis"] = dict(range=[0.0, r_max])

    fig.update_layout(**layout)
    return fig


def rdf_csv_bytes() -> bytes:
    """Export accumulated RDF counts as CSV bytes."""
    with st.session_state.worker_lock:
        bins = st.session_state.rdf_bins_edges
        accum = st.session_state.rdf_accum
        samples = int(st.session_state.rdf_samples)
        if bins is None or not accum or samples == 0:
            raise ValueError("No RDF data to export yet.")

        pairs = sorted(accum.keys())
        centers = 0.5 * (bins[:-1] + bins[1:])
        nbins = len(centers)

        buf = io.StringIO()
        w = csv.writer(buf)
        header = ["bin", "r_left", "r_right", "r_center"] + [
            f"{a}-{b}" for (a, b) in pairs
        ]
        w.writerow(header)
        for k in range(nbins):
            row = [k, float(bins[k]), float(bins[k + 1]), float(centers[k])]
            row.extend([int(accum[(a, b)][k]) for (a, b) in pairs])
            w.writerow(row)

    return buf.getvalue().encode("utf-8")


def rdf_save_csv_to_path(path: str) -> None:
    """Save accumulated RDF CSV to a file path."""
    data = rdf_csv_bytes()
    with open(path, "wb") as f:
        f.write(data)


# --------------------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------------------
def apply_plot_persistence(fig: go.Figure) -> None:
    """Set Plotly uirevision to persist camera/zoom across reruns."""
    fig.update_layout(uirevision="molmc")
    if hasattr(fig.layout, "scene") and fig.layout.scene is not None:
        fig.layout.scene.uirevision = "molmc"


def update_and_render(plot_area) -> None:
    """Update 3D scene traces and render exactly once per rerun."""
    if st.session_state.update_fig is not None and st.session_state.sim is not None:
        with st.session_state.worker_lock:
            _ = st.session_state.update_fig(st.session_state.sim)

    fig = st.session_state.fig
    if fig is None:
        return

    # (optional) reapply last camera when you *do* rebuild on purpose elsewhere
    if st.session_state.apply_camera_next and st.session_state.scene_camera is not None:
        fig.update_layout(scene_camera=st.session_state.scene_camera)
        st.session_state.apply_camera_next = False

    # Make sure scene-level uirevision is set (you already do this).
    if getattr(fig.layout, "scene", None) is not None:
        fig.layout.scene.uirevision = "molmc"

    plot_area.plotly_chart(
        fig,
        use_container_width=True,
        key="__main_plot__",
        config={"scrollZoom": True},
    )

    # Cache camera after render
    try:
        cam = fig.layout.scene.camera
        if cam:
            st.session_state.scene_camera = dict(cam)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# UI sections (sidebar & main)
# --------------------------------------------------------------------------------------
def render_quickstart(main_container) -> None:
    """Render collapsible quickstart/help panel."""
    with main_container.expander(
        "Quickstart & Help", expanded=st.session_state.show_quickstart
    ):
        st.markdown(
            """
**What to do**
1. **Initialize** a system in *Initialize / Load* (Gaussian files, counts, box, T).
2. Press **Start** in *Autoplay* to run MC in the background.
3. Adjust **Mutation coefficients** live to target your acceptance rates.
4. Let it run to approach equilibrium.
5. Use **RDF Recorder** → *Start* to accumulate snapshots (tune *stride* & *bins*).
6. **Export** RDF via *Download CSV* or save the simulation state.

**Tips**
- Manual stepping runs once per click and then redraws.
- Autoplay + Auto-sample ⇒ RDF snapshots tied to the step count.
            """
        )
        st.session_state.show_quickstart = st.checkbox(
            "Show this panel on startup", value=st.session_state.show_quickstart
        )


def sidebar_display_controls() -> None:
    """Render display cadence controls."""
    st.header("Display")
    st.number_input(
        "Redraw every (seconds)",
        min_value=0.05,
        max_value=10.0,
        step=0.05,
        key="redraw_interval_s",
        help="UI reruns only at this cadence; compute runs in background.",
    )


def sidebar_autoplay_controls() -> None:
    """Render autoplay controls and batch size input."""
    st.divider()
    st.header("Autoplay (background)")
    st.caption("Compute runs in a background thread; UI refresh is timer-driven.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start", use_container_width=True, type="primary"):
            start_worker()
    with c2:
        if st.button("Stop", use_container_width=True):
            stop_worker()
    st.number_input(
        "Worker batch size (steps/cycle)",
        min_value=1,
        max_value=100000,
        step=100,
        key="worker_batch",
        help="Higher = more steps per background cycle.",
    )


def sidebar_manual_stepping() -> None:
    """Render manual stepping controls and execute on click."""
    st.divider()
    st.header("Manual stepping")
    st.number_input(
        "Steps per redraw (manual)",
        min_value=1,
        max_value=1_000_000,
        step=50,
        key="manual_steps",
        help="When not using autoplay, run this many steps then redraw once.",
    )
    if st.button("Run steps, then redraw", use_container_width=True):
        if st.session_state.initialized:
            with st.session_state.worker_lock:
                for _ in range(st.session_state.manual_steps):
                    step_once(
                        st.session_state.sim,
                        st.session_state.forces,
                        st.session_state.rng,
                        st.session_state.mutation_functions,
                        st.session_state.function_coefficients,
                        st.session_state.recent_acceptance,
                        float(st.session_state.alpha),
                        st.session_state.stats,
                    )


def sidebar_mutation_coefficients() -> None:
    """Render editable mutation coefficients and apply."""
    st.divider()
    st.header("Mutation mix / coefficients")

    muts = st.session_state.mutation_functions
    coefs = st.session_state.function_coefficients
    rows = [
        {
            "Mutation": getattr(fn, "__name__", str(fn)),
            "Coefficient": float(coefs[i]) if i < len(coefs) else 1.0,
        }
        for i, fn in enumerate(muts)
    ]

    edited = st.data_editor(
        rows,
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Mutation": st.column_config.TextColumn(disabled=True),
            "Coefficient": st.column_config.NumberColumn(
                min_value=0.0, step=0.05, format="%.4f"
            ),
        },
        key="__coef_editor__",
    )

    if st.button("Apply", use_container_width=True):
        new = np.array([max(0.0, float(r["Coefficient"])) for r in edited], dtype=float)
        with st.session_state.worker_lock:
            st.session_state.function_coefficients = new
            if st.session_state.coef_reset_ema_on_apply or len(
                st.session_state.recent_acceptance
            ) != len(new):
                st.session_state.recent_acceptance = np.ones(len(new), dtype=float)
        st.success("Updated mutation coefficients.")


def _assign_new_system(sim, props, forces, rng, fig, updater, model) -> None:
    """Attach new system/figure to session_state and reset counters."""
    st.session_state.sim = sim
    st.session_state.props = props
    st.session_state.forces = forces
    st.session_state.rng = rng
    st.session_state.model = model
    st.session_state.fig = fig
    st.session_state.update_fig = updater
    st.session_state.initialized = True
    st.session_state.stats = {"accepted": 0, "trials": 0}
    st.session_state.recent_acceptance = np.ones(
        len(st.session_state.mutation_functions), dtype=float
    )
    st.session_state.apply_camera_next = True


def sidebar_initialize_or_load() -> None:
    """Render builders/loaders and assign the resulting system."""
    st.divider()
    st.header("Initialize / Load")

    st.subheader("Build system from Gaussian inputs")
    files_text = st.text_input(
        "Files (.com/.gjf, comma-separated)", value="resources/h2o.com"
    )
    numbers_text = st.text_input("Counts (integers, comma-separated)", value="10")
    box_text = st.text_input("Box (Å, 'Lx,Ly,Lz')", value="30,30,30")
    temperature = st.number_input("Temperature (K)", min_value=0.0, value=298.15)
    seed = st.number_input(
        "Seed", min_value=0, value=int(np.random.randint(int(1e6), size=1)), step=1
    )
    periodic = st.toggle(
        "Use PBC", value=True, help="Use periodic coundary conditions when on."
    )

    if st.button("Build system", use_container_width=True):
        try:
            files = [s.strip() for s in files_text.split(",") if s.strip()]
            numbers = [int(s.strip()) for s in numbers_text.split(",") if s.strip()]
            if len(files) != len(numbers):
                raise ValueError(
                    f"files ({len(files)}) and counts ({len(numbers)}) differ"
                )
            box = [float(s.strip()) for s in box_text.split(",")]
            if len(box) != 3:
                raise ValueError("box must have 3 values (Lx,Ly,Lz)")

            stop_worker()
            sim, props, forces, rng, fig, updater, model = build_system(
                files, numbers, box, temperature, seed, periodic
            )
            _assign_new_system(sim, props, forces, rng, fig, updater, model)
        except Exception as e:
            st.error(f"Failed to build system: {e}")

    st.subheader("Load from pickle")
    up = st.file_uploader("Upload state (.pickle/.pkl)", type=["pickle", "pkl"])
    pick_path = st.text_input("Or load from path", value="")
    c_up, c_path = st.columns(2)

    with c_up:
        if st.button("Load uploaded", use_container_width=True, disabled=(up is None)):
            try:
                stop_worker()
                obj = pickle.load(up)
                st.session_state.sim = obj
                st.session_state.props = getattr(obj, "properties", None)
                st.session_state.model = getattr(obj, "model", None)
                fig, updater = plot_system(st.session_state.sim)
                st.session_state.rng = RNG(seed)
                coulomb_cont = make_potcall(coulomb, st.session_state.sim.properties)
                lj_cont = make_potcall(lennard_jones, st.session_state.sim.properties)
                forces = SystemPotentials([coulomb_cont, lj_cont])
                _assign_new_system(
                    obj,
                    st.session_state.props,
                    forces,
                    st.session_state.rng,
                    fig,
                    updater,
                    st.session_state.model,
                )
            except Exception as e:
                st.error(f"Failed to load pickle: {e}")

    with c_path:
        if st.button(
            "Load from path",
            use_container_width=True,
            disabled=(pick_path.strip() == ""),
        ):
            try:
                stop_worker()
                with open(pick_path, "rb") as f:
                    obj = pickle.load(f)  # ⚠ trusted pickles only
                st.session_state.sim = obj
                st.session_state.props = getattr(obj, "properties", None)
                st.session_state.model = getattr(obj, "model", None)
                fig, updater = plot_system(st.session_state.sim)
                st.session_state.rng = RNG(seed)
                coulomb_cont = make_potcall(coulomb, st.session_state.sim.properties)
                lj_cont = make_potcall(lennard_jones, st.session_state.sim.properties)
                forces = SystemPotentials([coulomb_cont, lj_cont])
                _assign_new_system(
                    obj,
                    st.session_state.props,
                    forces,
                    st.session_state.rng,
                    fig,
                    updater,
                    st.session_state.model,
                )
            except Exception as e:
                st.error(f"Failed to load {pick_path}: {e}")


def sidebar_save_state() -> None:
    """Render save/download controls for current state."""
    st.subheader("Save current state")
    st.session_state.save_filename = st.text_input(
        "Filename",
        value=st.session_state.save_filename,
        help="Server-side path for 'Save to path'.",
    )

    c_spath, c_sdl = st.columns(2)
    with c_spath:
        if st.button(
            "Save to path",
            use_container_width=True,
            disabled=not st.session_state.initialized,
        ):
            try:
                if not st.session_state.save_filename.strip():
                    raise ValueError("Filename is empty.")
                save_sim_to_path(st.session_state.save_filename.strip())
                st.success(f"Saved → {st.session_state.save_filename}")
            except Exception as e:
                st.error(f"Failed to save: {e}")

    with c_sdl:
        if st.session_state.initialized:
            try:
                dl_bytes = pickle_sim_bytes()
                st.download_button(
                    "Download pickle",
                    data=dl_bytes,
                    file_name=st.session_state.save_filename or "state.pickle",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Download failed: {e}")
        else:
            st.download_button(
                "Download pickle",
                data=b"",
                file_name="state.pickle",
                disabled=True,
                use_container_width=True,
            )


def render_sidebar() -> None:
    """Render the complete sidebar."""
    with st.sidebar:
        sidebar_display_controls()
        sidebar_autoplay_controls()
        sidebar_manual_stepping()
        sidebar_mutation_coefficients()
        sidebar_initialize_or_load()
        sidebar_save_state()


def render_status_line() -> None:
    """Render acceptance statistics line."""
    acc = st.session_state.stats.get("accepted", 0)
    trl = st.session_state.stats.get("trials", 0)
    rate = (acc / trl) if trl > 0 else 0.0
    st.markdown(
        f"**Accepted:** {acc} / {trl} &nbsp;&nbsp; "
        f"**Recent EMA:** {', '.join(f'{v:.3f}' for v in st.session_state.recent_acceptance)} &nbsp;&nbsp; "
        f"**Rate:** {rate:.3f}"
    )


def render_rdf_recorder() -> None:
    """Render RDF recorder panel and export controls."""
    st.divider()
    with st.expander("RDF Recorder (accumulates snapshots)", expanded=False):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

        with c1:
            if st.button(
                "Start",
                use_container_width=True,
                disabled=not st.session_state.initialized,
            ):
                st.session_state.rdf_recording = True
        with c2:
            if st.button("Stop", use_container_width=True):
                st.session_state.rdf_recording = False
        with c3:
            if st.button("Reset", use_container_width=True):
                st.session_state.rdf_accum = {}
                st.session_state.rdf_samples = 0
                st.session_state.rdf_bins_edges = None
                st.session_state.rdf_last_sample_step = int(
                    st.session_state.stats.get("trials", 0)
                )
        with c4:
            st.number_input(
                "Bins",
                min_value=10,
                max_value=5000,
                step=10,
                key="rdf_bins_count",
                help="Number of histogram bins for r (Å).",
            )

        c_stride, c_auto = st.columns([1, 1])
        with c_stride:
            st.number_input(
                "Sample stride (steps)",
                min_value=1,
                max_value=1_000_000,
                step=10,
                key="rdf_stride",
                help="Record one RDF snapshot every N MC steps.",
            )
        with c_auto:
            st.checkbox(
                "Auto-sample in worker",
                key="rdf_auto_sample",
                help="If off, a snapshot is taken on each UI redraw while recording.",
            )

        # Manual snapshot on rerun (when auto-sample is OFF)
        if (
            st.session_state.initialized
            and st.session_state.rdf_recording
            and not st.session_state.rdf_auto_sample
        ):
            try:
                with st.session_state.worker_lock:
                    rdf_record_single_snapshot(int(st.session_state.rdf_bins_count))
            except Exception as e:
                st.error(f"RDF recording failed: {e}")

        rdf_fig = rdf_plot()
        if rdf_fig is not None:
            st.plotly_chart(rdf_fig, use_container_width=True, key="__rdf_plot__")
            st.caption(
                f"Samples accumulated: {st.session_state.rdf_samples} · "
                f"Pairs: {len(st.session_state.rdf_accum)}"
            )
        else:
            st.caption("No RDF data yet. Click **Start** to begin accumulating.")

        st.markdown("**Export**")
        st.session_state.rdf_csv_filename = st.text_input(
            "CSV filename", value=st.session_state.rdf_csv_filename
        )
        c_csv_path, c_csv_dl = st.columns(2)
        with c_csv_path:
            if st.button(
                "Save CSV to path",
                use_container_width=True,
                disabled=(
                    st.session_state.rdf_bins_edges is None
                    or st.session_state.rdf_samples == 0
                ),
            ):
                try:
                    rdf_save_csv_to_path(st.session_state.rdf_csv_filename)
                    st.success(f"Saved → {st.session_state.rdf_csv_filename}")
                except Exception as e:
                    st.error(f"Failed to save CSV: {e}")

        with c_csv_dl:
            try:
                csv_bytes = rdf_csv_bytes()
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name=st.session_state.rdf_csv_filename or "rdf.csv",
                    mime="text/csv",
                    use_container_width=True,
                    disabled=(
                        st.session_state.rdf_bins_edges is None
                        or st.session_state.rdf_samples == 0
                    ),
                )
            except Exception:
                st.download_button(
                    "Download CSV",
                    data=b"",
                    file_name=st.session_state.rdf_csv_filename or "rdf.csv",
                    mime="text/csv",
                    use_container_width=True,
                    disabled=True,
                )


# --------------------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit app."""
    configure_page()
    ensure_state()

    # Layout columns
    main_col, right_col = st.columns([5, 2], gap="large")
    with main_col:
        plot_area = st.empty()

    with right_col:
        render_quickstart(st)

    render_sidebar()

    # One render per rerun
    if st.session_state.initialized and st.session_state.fig is not None:
        update_and_render(plot_area)

    render_status_line()
    render_rdf_recorder()

    # Non-blocking, browser-driven rerun
    if st.session_state.worker_running and st.session_state.initialized:
        st_autorefresh(
            interval=int(float(st.session_state.redraw_interval_s) * 1000),
            key="__auto_refresh__",
        )


if __name__ == "__main__":
    main()
