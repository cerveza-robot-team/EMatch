"""
interactive_align.py – Interactive event-rate alignment + HDF5 export.

Discovers ALL master/slave .raw pairs in a sequence directory and processes
them one by one. For each pair the user is prompted for an output name, then
shown a binned event-rate plot with two sliders:

  • Shift slider  – horizontal time offset applied to slave timestamps
  • Skip slider   – start-of-recording crop in the SHIFTED timeline
                    (one line covers both cameras after alignment)

Closing the window exports an MVSEC-style HDF5 file, then the next pair loads.

Usage:
    python interactive_align.py              # auto-select if single sequence found
    python interactive_align.py --seq new    # sequence name (subdir of raw/)
    python interactive_align.py --seq /full/path/to/seq_dir
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Slider, Button
from metavision_core.event_io import RawReader

# ── Paths ─────────────────────────────────────────────────────────────────────
EMATCH_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_ROOT = os.path.join(EMATCH_DIR, "data", "EVB_CIRS")
RAW_DIR      = os.path.join(DATASET_ROOT, "raw")  # sequence dirs (e.g. new/, old/) live here
HDF5_OUT_DIR = os.path.join(DATASET_ROOT, "hdf5")

CHUNK_SIZE        = 500_000
BIN_SIZE_US       = 10_000   # 10 ms per histogram bin
CHUNK_DURATION_US = 10_000   # 10 ms read window during HDF5 export


# ── Event-rate computation ─────────────────────────────────────────────────────
def compute_event_rate(file_path, bin_size_us=BIN_SIZE_US, chunk_size=CHUNK_SIZE):
    """Return (time_s, counts, total) for the binned event rate over the whole file."""
    reader = RawReader(file_path)
    reader.seek_time(0)
    bin_counts: dict[int, int] = {}
    total = 0
    while not reader.is_done():
        evs = reader.load_n_events(chunk_size)
        if evs.size == 0:
            continue
        bin_idx = evs["t"].astype(np.int64) // bin_size_us
        unique_bins, counts = np.unique(bin_idx, return_counts=True)
        for b, c in zip(unique_bins.tolist(), counts.tolist()):
            bin_counts[b] = bin_counts.get(b, 0) + c
        total += evs.size
    bins   = np.array(sorted(bin_counts.keys()), dtype=np.int64)
    counts = np.array([bin_counts[b] for b in bins], dtype=np.int64)
    time_s = bins * bin_size_us / 1e6
    return time_s, counts, total


# ── Raw → events array ────────────────────────────────────────────────────────
def read_raw_events(raw_path: str, time_skip_us: int, time_shift_us: int = 0) -> np.ndarray:
    """Read events starting at time_skip_us; optionally offset timestamps by time_shift_us.

    Returns (N, 4) float64 array: [x, y, timestamp_us, polarity (-1/+1)].
    """
    reader = RawReader(raw_path)
    reader.seek_time(time_skip_us)
    chunks = []
    while not reader.is_done():
        evs = reader.load_delta_t(CHUNK_DURATION_US)
        if evs is None or len(evs) == 0:
            continue
        arr = np.column_stack([
            evs["x"].astype(np.float64),
            evs["y"].astype(np.float64),
            evs["t"].astype(np.float64) + time_shift_us,
            evs["p"].astype(np.float64) * 2 - 1,   # 0/1 → -1/+1
        ])
        chunks.append(arr)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)


# ── Sequence discovery ────────────────────────────────────────────────────────
def find_all_raw_pairs(seq_dir: str):
    """Return a sorted list of (master_path, slave_path) tuples found in seq_dir.

    Masters and slaves are matched by sort order (first master with first slave, etc.).
    Warns if counts differ and pairs only the common prefix.
    """
    masters = sorted(glob.glob(os.path.join(seq_dir, "*cam_master_*.raw")))
    slaves  = sorted(glob.glob(os.path.join(seq_dir, "*cam_slave_*.raw")))
    if not masters:
        raise FileNotFoundError(f"No master .raw files in {seq_dir}")
    if not slaves:
        raise FileNotFoundError(f"No slave .raw files in {seq_dir}")
    if len(masters) != len(slaves):
        print(f"WARNING: found {len(masters)} master(s) and {len(slaves)} slave(s) — "
              f"pairing the first {min(len(masters), len(slaves))}.", file=sys.stderr)
    n = min(len(masters), len(slaves))
    return list(zip(masters[:n], slaves[:n]))


def discover_sequences():
    if not os.path.isdir(RAW_DIR):
        return []
    seqs = []
    for entry in sorted(os.listdir(RAW_DIR)):
        full = os.path.join(RAW_DIR, entry)
        if os.path.isdir(full) and not entry.startswith("."):
            m = glob.glob(os.path.join(full, "*cam_master_*.raw"))
            s = glob.glob(os.path.join(full, "*cam_slave_*.raw"))
            if m and s:
                seqs.append(full)
    return seqs


# ── Per-pair interactive alignment ────────────────────────────────────────────
def process_pair(seq_name: str, master_path: str, slave_path: str, out_name: str):
    """Show the interactive alignment window for one master/slave pair and export on close."""

    # ── Load event rates ──────────────────────────────────────────────────────
    print("\n[MASTER] Computing event rate ...")
    mt, mc, m_total = compute_event_rate(master_path)
    print(f"  events : {m_total:,}  |  {mt[0]:.3f} s → {mt[-1]:.3f} s")

    print("[SLAVE]  Computing event rate ...")
    st, sc, s_total = compute_event_rate(slave_path)
    print(f"  events : {s_total:,}  |  {st[0]:.3f} s → {st[-1]:.3f} s")

    # ── Initial estimates ─────────────────────────────────────────────────────
    first_shift = mt[0]  - st[0]
    last_shift  = mt[-1] - st[-1]   # align by last event — more reliable
    init_shift  = last_shift
    init_skip   = 0.0

    shift_range = max(abs(last_shift), abs(first_shift)) + 2.0
    skip_max    = min(mt[-1] - mt[0], st[-1] - st[0]) * 0.5

    print(f"\nShift estimate (first event) : {first_shift*1e6:+.2f} µs")
    print(f"Shift estimate (last  event) : {last_shift*1e6:+.2f} µs  ← initial")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7))
    plt.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.30)

    line_m, = ax.plot(mt, mc,
                      color="steelblue",  lw=0.8, alpha=0.85, label="MASTER")
    line_s, = ax.plot(st + init_shift, sc,
                      color="darkorange", lw=0.8, alpha=0.85, label="SLAVE (shifted)")

    # Single dashed vertical line at the skip cut-off (shared by both cameras
    # because the slave is already shifted into the master timeline).
    vline_skip = ax.axvline(init_skip, color="black",
                            ls="--", lw=1.2, alpha=0.7, label="Skip")

    # Grey shaded region covering discarded (pre-skip) events.
    trans      = blended_transform_factory(ax.transData, ax.transAxes)
    x_far_left = min(mt[0], st[0] + init_shift) - 10.0
    skip_rect  = Rectangle(
        (x_far_left, 0), init_skip - x_far_left, 1.0,
        transform=trans, color="gray", alpha=0.12, zorder=0
    )
    ax.add_patch(skip_rect)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Events per {BIN_SIZE_US // 1000} ms bin")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    title_obj = ax.set_title(
        f"{seq_name} / {out_name}  |  shift: {init_shift*1e6:+.1f} µs   skip: {init_skip*1e3:.1f} ms",
        fontsize=11
    )

    # ── Sliders ───────────────────────────────────────────────────────────────
    ax_shift = plt.axes([0.12, 0.18, 0.76, 0.035])
    ax_skip  = plt.axes([0.12, 0.10, 0.76, 0.035])

    sl_shift = Slider(ax_shift, "Shift (s)", -shift_range, shift_range,
                      valinit=init_shift, color="darkorange")
    sl_skip  = Slider(ax_skip,  "Skip  (s)", 0.0, skip_max,
                      valinit=init_skip,  color="steelblue")

    ax_reset  = plt.axes([0.905, 0.04, 0.07, 0.04])
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.85")

    fig.text(0.12, 0.04,
             f"Shift ±{shift_range:.3f} s  |  Skip 0 – {skip_max:.2f} s  |  "
             f"bin {BIN_SIZE_US // 1000} ms  |  Close window to export HDF5",
             fontsize=8, color="gray")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def update(_val):
        shift = sl_shift.val
        skip  = sl_skip.val

        line_s.set_xdata(st + shift)
        vline_skip.set_xdata([skip, skip])

        new_x_left = min(mt[0], st[0] + shift) - 10.0
        skip_rect.set_x(new_x_left)
        skip_rect.set_width(skip - new_x_left)

        title_obj.set_text(
            f"{seq_name} / {out_name}  |  shift: {shift*1e6:+.1f} µs   skip: {skip*1e3:.1f} ms"
        )
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def reset(_event):
        sl_shift.reset()
        sl_skip.reset()

    sl_shift.on_changed(update)
    sl_skip.on_changed(update)
    btn_reset.on_clicked(reset)

    # ── On close: export HDF5 ─────────────────────────────────────────────────
    def on_close(_event):
        shift_s  = sl_shift.val
        skip_s   = sl_skip.val
        shift_us = int(round(shift_s * 1e6))
        skip_us  = int(round(skip_s  * 1e6))

        print(f"\n[RESULT]  shift = {shift_us:+d} µs   skip = {skip_us} µs")

        os.makedirs(HDF5_OUT_DIR, exist_ok=True)
        out_path = os.path.join(HDF5_OUT_DIR, f"{out_name}.hdf5")

        print(f"[EXPORT]  Reading master (skip={skip_us} µs) ...")
        left_events = read_raw_events(master_path, time_skip_us=skip_us)
        print(f"          {len(left_events):,} events")

        # Seek slave in raw time: skip_us - shift_us → after adding shift_us,
        # events start at skip_us (same as master).
        slave_seek_us = max(0, skip_us - shift_us)
        print(f"[EXPORT]  Reading slave  (raw seek={slave_seek_us} µs, shift={shift_us:+d} µs) ...")
        right_events = read_raw_events(slave_path,
                                       time_skip_us=slave_seek_us,
                                       time_shift_us=shift_us)
        print(f"          {len(right_events):,} events")

        print(f"[EXPORT]  Writing {out_path} ...")
        with h5py.File(out_path, "w") as f:
            grp = f.create_group("evk4_hd")
            grp.create_dataset("left/events",  data=left_events,
                               compression="gzip", compression_opts=4)
            grp.create_dataset("right/events", data=right_events,
                               compression="gzip", compression_opts=4)

        png_path = os.path.join(HDF5_OUT_DIR, f"{out_name}_alignment.png")
        fig.savefig(png_path, dpi=150)
        print(f"[EXPORT]  Done.\n  HDF5 → {out_path}\n  Plot → {png_path}")

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Interactive alignment + HDF5 export")
    parser.add_argument("--seq", metavar="NAME_OR_PATH",
                        help="Sequence name (subdir of raw/) or absolute directory path")
    args = parser.parse_args()

    # ── Resolve sequence directory ────────────────────────────────────────────
    if args.seq:
        candidate = args.seq if os.path.isabs(args.seq) else os.path.join(RAW_DIR, args.seq)
        if not os.path.isdir(candidate):
            print(f"ERROR: '{candidate}' is not a directory.", file=sys.stderr)
            sys.exit(1)
        seq_dir = candidate
    else:
        seqs = discover_sequences()
        if not seqs:
            print(f"No sequence dirs with master+slave .raw files found under: {RAW_DIR}",
                  file=sys.stderr)
            sys.exit(1)
        if len(seqs) == 1:
            seq_dir = seqs[0]
            print(f"Auto-selected: {seq_dir}")
        else:
            print("Multiple sequences found — select one with --seq <name>:")
            for s in seqs:
                print(f"  {os.path.basename(s)}")
            sys.exit(0)

    seq_name = os.path.basename(seq_dir)
    pairs    = find_all_raw_pairs(seq_dir)
    n_pairs  = len(pairs)
    print(f"Sequence : {seq_name}  ({n_pairs} pair{'s' if n_pairs != 1 else ''} found)")

    # ── Process each pair ─────────────────────────────────────────────────────
    for i, (master_path, slave_path) in enumerate(pairs):
        print(f"\n{'─'*60}")
        print(f"Pair {i+1}/{n_pairs}")
        print(f"  master : {os.path.basename(master_path)}")
        print(f"  slave  : {os.path.basename(slave_path)}")

        stem = os.path.splitext(os.path.basename(master_path))[0]
        default_name = stem.replace("cam_master_", "").strip("_- ")
        user_input = input(f"Output name [{default_name}]: ").strip()
        out_name = user_input if user_input else default_name

        process_pair(seq_name, master_path, slave_path, out_name)

    print(f"\nAll {n_pairs} pair{'s' if n_pairs != 1 else ''} processed.")


if __name__ == "__main__":
    main()
