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
import ctypes
import ctypes.util
import gc
import glob
import os
import resource
import sys

try:
    _LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    _MALLOC_TRIM = _LIBC.malloc_trim
    _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _MALLOC_TRIM.restype = ctypes.c_int
except (OSError, AttributeError):
    _MALLOC_TRIM = None


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _trim_memory():
    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)

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
MAX_EVENTS        = 20_000_000  # RawReader internal buffer — must be ≥ events in one decode batch
BIN_SIZE_US       = 10_000   # 10 ms per histogram bin
CHUNK_DURATION_US = 10_000   # 10 ms read window during HDF5 export
WRITE_BATCH_EVENTS = 1_000_000  # accumulate before flushing to HDF5 (fewer resize calls)


# ── Event-rate computation ─────────────────────────────────────────────────────
def compute_event_rate(file_path, bin_size_us=BIN_SIZE_US, chunk_size=CHUNK_SIZE):
    """Return (time_s, counts, total) for the binned event rate over the whole file."""
    reader = RawReader(file_path, max_events=MAX_EVENTS)
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


# ── Raw → HDF5 streaming writer ──────────────────────────────────────────────
FLUSH_EVERY = 10_000_000   # flush + progress print every N events written


def stream_raw_to_hdf5(raw_path: str, h5_group, dataset_name: str,
                       time_skip_us: int, time_shift_us: int = 0,
                       time_end_us: int = None) -> int:
    """Stream events from a raw file directly to a resizable HDF5 dataset.

    Only one ~10 ms decode chunk is held in memory at a time, so this handles
    arbitrarily long / dense recordings without OOM crashes.

    Compression is intentionally disabled: gzip compression + chunk cache for
    a 2+ GB dataset can use several GB of RAM on top of the raw data, which
    causes OOM kills on dense multi-minute recordings. The output file is
    larger on disk but the process stays within ~500 MB RAM.

    time_end_us is in the OUTPUT (shifted) timestamp space — same as the master timeline.
    Returns the total number of events written.
    """
    reader = RawReader(raw_path, max_events=MAX_EVENTS)
    reader.seek_time(time_skip_us)

    dset = h5_group.create_dataset(
        dataset_name,
        shape=(0, 4), maxshape=(None, 4), dtype=np.float64,
        chunks=(100_000, 4),   # no compression — see docstring
    )

    buf, buf_n = [], 0
    total = 0

    def flush_buf():
        nonlocal buf, buf_n, total
        if buf_n == 0:
            return
        batch = np.concatenate(buf, axis=0)
        dset.resize((total + buf_n, 4))
        dset[total:total + buf_n] = batch
        total += buf_n
        buf, buf_n = [], 0
        del batch
        h5_group.file.flush()
        _trim_memory()
        print(f"            ... {total:,} events written  RSS={_rss_mb():.0f} MB",
              flush=True)

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
        past_end = False
        if time_end_us is not None:
            arr = arr[arr[:, 2] <= time_end_us]
            past_end = (len(arr) == 0)   # fully past the cutoff → stop

        n = len(arr)
        if n > 0:
            buf.append(arr)
            buf_n += n

        if buf_n >= WRITE_BATCH_EVENTS:
            flush_buf()

        if past_end:
            break

    flush_buf()
    h5_group.file.flush()
    del reader
    _trim_memory()
    return total


# ── Sequence discovery ────────────────────────────────────────────────────────
def find_all_raw_pairs(seq_dir: str):
    """Return a sorted list of (master_path, slave_path) tuples found in seq_dir.

    Matching strategy:
      1. For each master, try to find its slave by replacing 'master' → 'slave'
         in the filename (handles e.g. '1m_master.raw' → '1m_slave.raw').
      2. Any unmatched files fall back to sort-order pairing.
    """
    masters = sorted(glob.glob(os.path.join(seq_dir, "*master*.raw")))
    slaves  = sorted(glob.glob(os.path.join(seq_dir, "*slave*.raw")))
    if not masters:
        raise FileNotFoundError(f"No master .raw files in {seq_dir}")
    if not slaves:
        raise FileNotFoundError(f"No slave .raw files in {seq_dir}")

    slave_by_name = {os.path.basename(s): s for s in slaves}
    paired, unmatched_masters, used_slaves = [], [], set()

    for m in masters:
        stem       = os.path.basename(m)
        slave_stem = stem.replace("master", "slave")
        if slave_stem in slave_by_name:
            paired.append((m, slave_by_name[slave_stem]))
            used_slaves.add(slave_stem)
        else:
            unmatched_masters.append(m)

    unmatched_slaves = [s for s in slaves if os.path.basename(s) not in used_slaves]
    n = min(len(unmatched_masters), len(unmatched_slaves))
    if n:
        print(f"WARNING: {n} pair(s) matched by sort order (no name match found).",
              file=sys.stderr)
    paired += list(zip(unmatched_masters[:n], unmatched_slaves[:n]))

    return sorted(paired)


def discover_sequences():
    if not os.path.isdir(RAW_DIR):
        return []
    seqs = []
    for entry in sorted(os.listdir(RAW_DIR)):
        full = os.path.join(RAW_DIR, entry)
        if os.path.isdir(full) and not entry.startswith("."):
            m = glob.glob(os.path.join(full, "*_master_*.raw"))
            s = glob.glob(os.path.join(full, "*_slave_*.raw"))
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
    init_skip     = 0.0
    init_skip_end = 0.0

    shift_range   = max(abs(last_shift), abs(first_shift)) + 2.0
    skip_max      = min(mt[-1] - mt[0], st[-1] - st[0]) * 0.5
    # t_max: latest point in the master timeline where both cameras still have data
    t_max         = min(mt[-1], st[-1] + init_shift)

    print(f"\nShift estimate (first event) : {first_shift*1e6:+.2f} µs")
    print(f"Shift estimate (last  event) : {last_shift*1e6:+.2f} µs  ← initial")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7))
    plt.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.38)

    line_m, = ax.plot(mt, mc,
                      color="steelblue",  lw=0.8, alpha=0.85, label="MASTER")
    line_s, = ax.plot(st + init_shift, sc,
                      color="darkorange", lw=0.8, alpha=0.85, label="SLAVE (shifted)")

    # Dashed vertical lines at the start and end crop boundaries.
    vline_skip = ax.axvline(init_skip, color="black",
                            ls="--", lw=1.2, alpha=0.7, label="Skip start")
    vline_end  = ax.axvline(t_max - init_skip_end, color="darkred",
                            ls="--", lw=1.2, alpha=0.7, label="Skip end")

    # Grey shaded regions covering discarded events.
    trans       = blended_transform_factory(ax.transData, ax.transAxes)
    x_far_left  = min(mt[0], st[0] + init_shift) - 10.0
    x_far_right = max(mt[-1], st[-1] + init_shift) + 10.0
    skip_rect = Rectangle(
        (x_far_left, 0), init_skip - x_far_left, 1.0,
        transform=trans, color="gray", alpha=0.12, zorder=0
    )
    end_rect = Rectangle(
        (t_max - init_skip_end, 0), x_far_right - (t_max - init_skip_end), 1.0,
        transform=trans, color="gray", alpha=0.12, zorder=0
    )
    ax.add_patch(skip_rect)
    ax.add_patch(end_rect)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Events per {BIN_SIZE_US // 1000} ms bin")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    title_obj = ax.set_title(
        f"{seq_name} / {out_name}  |  shift: {init_shift*1e6:+.1f} µs"
        f"   skip: {init_skip*1e3:.1f} ms   skip_end: {init_skip_end*1e3:.1f} ms",
        fontsize=11
    )

    # ── Sliders ───────────────────────────────────────────────────────────────
    ax_shift    = plt.axes([0.12, 0.26, 0.76, 0.035])
    ax_skip     = plt.axes([0.12, 0.18, 0.76, 0.035])
    ax_skip_end = plt.axes([0.12, 0.10, 0.76, 0.035])

    sl_shift    = Slider(ax_shift,    "Shift (s)",    -shift_range, shift_range,
                         valinit=init_shift,    color="darkorange")
    sl_skip     = Slider(ax_skip,     "Skip start (s)", 0.0, skip_max,
                         valinit=init_skip,     color="steelblue")
    sl_skip_end = Slider(ax_skip_end, "Skip end   (s)", 0.0, skip_max,
                         valinit=init_skip_end, color="darkred")

    ax_reset  = plt.axes([0.905, 0.04, 0.07, 0.04])
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.85")

    fig.text(0.12, 0.04,
             f"Shift ±{shift_range:.3f} s  |  Skip 0 – {skip_max:.2f} s  |  "
             f"bin {BIN_SIZE_US // 1000} ms  |  Close window to export HDF5",
             fontsize=8, color="gray")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def update(_val):
        shift    = sl_shift.val
        skip     = sl_skip.val
        skip_end = sl_skip_end.val
        t_max_now = min(mt[-1], st[-1] + shift)

        line_s.set_xdata(st + shift)
        vline_skip.set_xdata([skip, skip])
        vline_end.set_xdata([t_max_now - skip_end, t_max_now - skip_end])

        new_x_left = min(mt[0], st[0] + shift) - 10.0
        skip_rect.set_x(new_x_left)
        skip_rect.set_width(skip - new_x_left)
        end_rect.set_x(t_max_now - skip_end)
        end_rect.set_width(x_far_right - (t_max_now - skip_end))

        title_obj.set_text(
            f"{seq_name} / {out_name}  |  shift: {shift*1e6:+.1f} µs"
            f"   skip: {skip*1e3:.1f} ms   skip_end: {skip_end*1e3:.1f} ms"
        )
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def reset(_event):
        sl_shift.reset()
        sl_skip.reset()
        sl_skip_end.reset()

    sl_shift.on_changed(update)
    sl_skip.on_changed(update)
    sl_skip_end.on_changed(update)
    btn_reset.on_clicked(reset)

    # ── On close: export HDF5 ─────────────────────────────────────────────────
    def on_close(_event):
        import traceback
        crash_log = os.path.join(HDF5_OUT_DIR, f"{out_name}_export_error.txt")
        try:
            shift_s    = sl_shift.val
            skip_s     = sl_skip.val
            skip_end_s = sl_skip_end.val
            shift_us   = int(round(shift_s    * 1e6))
            skip_us    = int(round(skip_s     * 1e6))
            t_end_us   = int(round((min(mt[-1], st[-1] + shift_s) - skip_end_s) * 1e6))

            print(f"\n[RESULT]  shift = {shift_us:+d} µs   skip = {skip_us} µs"
                  f"   t_end = {t_end_us} µs  (skip_end = {int(skip_end_s*1e3)} ms)",
                  flush=True)

            os.makedirs(HDF5_OUT_DIR, exist_ok=True)
            out_path = os.path.join(HDF5_OUT_DIR, f"{out_name}.hdf5")

            slave_seek_us = max(0, skip_us - shift_us)
            print(f"[EXPORT]  Writing {out_path} ...", flush=True)
            with h5py.File(out_path, "w") as f:
                grp = f.create_group("evk4_hd")
                print(f"          master (skip={skip_us} µs, t_end={t_end_us} µs) ...",
                      flush=True)
                n_left = stream_raw_to_hdf5(master_path, grp, "left/events",
                                            time_skip_us=skip_us, time_end_us=t_end_us)
                print(f"          {n_left:,} events", flush=True)
                print(f"          slave  (raw seek={slave_seek_us} µs, shift={shift_us:+d} µs) ...",
                      flush=True)
                n_right = stream_raw_to_hdf5(slave_path, grp, "right/events",
                                             time_skip_us=slave_seek_us,
                                             time_shift_us=shift_us,
                                             time_end_us=t_end_us)
                print(f"          {n_right:,} events", flush=True)

            png_path = os.path.join(HDF5_OUT_DIR, f"{out_name}_alignment.png")
            fig.savefig(png_path, dpi=150)
            print(f"[EXPORT]  Done.\n  HDF5 → {out_path}\n  Plot → {png_path}",
                  flush=True)
        except BaseException as e:
            tb = traceback.format_exc()
            print('\n[EXPORT][ERROR] ' + tb, flush=True)
            try:
                os.makedirs(HDF5_OUT_DIR, exist_ok=True)
                with open(crash_log, 'w') as fh:
                    fh.write(tb)
                print(f'[EXPORT][ERROR] traceback written to {crash_log}', flush=True)
            except Exception:
                pass
            raise

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()
    plt.close(fig)


# ── Pair selection ────────────────────────────────────────────────────────────
def parse_pair_selection(spec: str, pairs):
    """Parse a selection string (e.g. '0,2,4-6' or '1m,2m_on') into a list of pair indices.

    Accepts:
      - 'all' / '' → every pair
      - comma-separated indices : '0,2,4'
      - ranges                  : '1-3'   (inclusive on both ends)
      - master-file name stems  : '1m,2m_on'  (matches if basename contains the token)
    """
    spec = (spec or "").strip().lower()
    if not spec or spec == "all":
        return list(range(len(pairs)))

    selected = []
    master_names = [os.path.basename(m).lower() for m, _ in pairs]
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token.isdigit():
            i = int(token)
            if 0 <= i < len(pairs):
                selected.append(i)
            else:
                print(f"WARNING: index {i} out of range — ignored.", file=sys.stderr)
        elif "-" in token and all(p.isdigit() for p in token.split("-")):
            a, b = (int(p) for p in token.split("-"))
            selected.extend(range(max(0, a), min(len(pairs), b + 1)))
        else:
            matches = [i for i, name in enumerate(master_names) if token in name]
            if matches:
                selected.extend(matches)
            else:
                print(f"WARNING: no pair matches token '{token}' — ignored.", file=sys.stderr)
    # Dedupe while preserving order
    seen, unique = set(), []
    for i in selected:
        if i not in seen:
            unique.append(i); seen.add(i)
    return unique


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Interactive alignment + HDF5 export")
    parser.add_argument("--seq", metavar="NAME_OR_PATH",
                        help="Sequence name (subdir of raw/) or absolute directory path")
    parser.add_argument("--pair", metavar="SPEC",
                        help="Select specific pair(s): 'all', indices ('0,2,4-6'), "
                             "or filename tokens ('1m,2m_on'). If omitted, you'll be "
                             "prompted interactively.")
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

    # ── List and select pairs ─────────────────────────────────────────────────
    print(f"\nAvailable pairs:")
    for i, (m, s) in enumerate(pairs):
        print(f"  [{i}] {os.path.basename(m)}  ↔  {os.path.basename(s)}")

    if args.pair is not None:
        spec = args.pair
    else:
        spec = input("\nProcess which? (indices '0,2,4-6' | tokens '1m,2m' | Enter=all): ").strip()
    selected = parse_pair_selection(spec, pairs)
    if not selected:
        print("No pairs selected — exiting.", file=sys.stderr)
        sys.exit(0)
    print(f"\nSelected {len(selected)} pair(s): {selected}")

    # ── Process each selected pair ────────────────────────────────────────────
    for n, idx in enumerate(selected, start=1):
        master_path, slave_path = pairs[idx]
        print(f"\n{'─'*60}")
        print(f"Pair {n}/{len(selected)}  (index {idx})")
        print(f"  master : {os.path.basename(master_path)}")
        print(f"  slave  : {os.path.basename(slave_path)}")

        stem = os.path.splitext(os.path.basename(master_path))[0]
        default_name = stem.replace("_master_", "").replace("_master", "").strip("_- ")
        user_input = input(f"Output name [{default_name}]: ").strip()
        out_name = user_input if user_input else default_name

        process_pair(seq_name, master_path, slave_path, out_name)

    print(f"\nAll {len(selected)} pair(s) processed.")


if __name__ == "__main__":
    main()
