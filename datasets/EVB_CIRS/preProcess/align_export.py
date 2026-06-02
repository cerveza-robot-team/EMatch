"""Non-interactive HDF5 exporter for an aligned master/slave .raw pair.

Takes the alignment values you would have set via interactive_align.py's
sliders and streams events to MVSEC-style HDF5. No matplotlib, no GUI — runs
purely in a terminal so it survives X-server / GUI-thread instability.

Usage:
    python align_export.py \
        --master raw/indoor_dataset/1_5m_master.raw \
        --slave  raw/indoor_dataset/1_5_slave.raw  \
        --output 1_5m_indoor \
        --shift -9.27 --skip 1.6 --skip_end 0.642
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import h5py
import numpy as np
from metavision_core.event_io import RawReader

EMATCH_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_ROOT = os.path.join(EMATCH_DIR, "data", "EVB_CIRS")
HDF5_OUT_DIR = os.path.join(DATASET_ROOT, "hdf5")

CHUNK_DURATION_US = 10_000
FLUSH_EVERY = 10_000_000
MAX_EVENTS = 100_000_000


def stream_raw_to_hdf5(raw_path, h5_group, dataset_name,
                       time_skip_us, time_shift_us=0, time_end_us=None):
    reader = RawReader(raw_path, max_events=MAX_EVENTS)
    reader.seek_time(time_skip_us)

    dset = h5_group.create_dataset(
        dataset_name,
        shape=(0, 4), maxshape=(None, 4), dtype=np.float64,
        chunks=(100_000, 4),
    )

    total, last_flush = 0, 0
    while not reader.is_done():
        evs = reader.load_delta_t(CHUNK_DURATION_US)
        if evs is None or len(evs) == 0:
            continue
        arr = np.column_stack([
            evs["x"].astype(np.float64),
            evs["y"].astype(np.float64),
            evs["t"].astype(np.float64) + time_shift_us,
            evs["p"].astype(np.float64) * 2 - 1,
        ])
        past_end = False
        if time_end_us is not None:
            arr = arr[arr[:, 2] <= time_end_us]
            past_end = (len(arr) == 0)

        n = len(arr)
        if n > 0:
            dset.resize((total + n, 4))
            dset[total:total + n] = arr
            total += n

        if total - last_flush >= FLUSH_EVERY:
            h5_group.file.flush()
            print(f"  ... {total:,} events written", flush=True)
            last_flush = total

        if past_end:
            break
    h5_group.file.flush()
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--master', required=True)
    ap.add_argument('--slave', required=True)
    ap.add_argument('--output', required=True,
                    help='output stem; final file = HDF5_OUT_DIR/<output>.hdf5')
    ap.add_argument('--out-dir', default=HDF5_OUT_DIR,
                    help=f'override output dir (default: {HDF5_OUT_DIR})')
    ap.add_argument('--shift', type=float, required=True,
                    help='slave time shift in seconds (added to slave timestamps to align with master)')
    ap.add_argument('--skip', type=float, default=0.0,
                    help='skip start, seconds (in shifted/master timeline)')
    ap.add_argument('--skip_end', type=float, default=0.0,
                    help='skip end, seconds (drop final N s)')
    ap.add_argument('--master_t_max', type=float, default=None,
                    help='master last event timestamp in seconds (auto-probed if omitted)')
    ap.add_argument('--slave_t_max', type=float, default=None,
                    help='slave last event timestamp in seconds (auto-probed if omitted)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.output}.hdf5')
    crash_log = os.path.join(args.out_dir, f'{args.output}_export_error.txt')

    if args.master_t_max is None or args.slave_t_max is None:
        print('[probe] reading master/slave end timestamps ...', flush=True)
        for tag, p in [('master', args.master), ('slave', args.slave)]:
            r = RawReader(p, max_events=MAX_EVENTS)
            r.seek_time(0)
            last_t = 0
            while not r.is_done():
                evs = r.load_n_events(1_000_000)
                if evs.size == 0:
                    continue
                last_t = int(evs['t'][-1])
            print(f'  {tag}: last_t = {last_t} us = {last_t/1e6:.3f} s', flush=True)
            if tag == 'master' and args.master_t_max is None:
                args.master_t_max = last_t / 1e6
            if tag == 'slave' and args.slave_t_max is None:
                args.slave_t_max = last_t / 1e6

    shift_us    = int(round(args.shift * 1e6))
    skip_us     = int(round(args.skip * 1e6))
    skip_end_us = int(round(args.skip_end * 1e6))
    t_max_aligned = min(args.master_t_max, args.slave_t_max + args.shift)
    t_end_us = int(round((t_max_aligned - args.skip_end) * 1e6))

    print(f'[plan] shift={shift_us:+d} us  skip={skip_us} us  '
          f't_end={t_end_us} us  (skip_end={skip_end_us} us)', flush=True)
    print(f'[plan] writing -> {out_path}', flush=True)

    slave_seek_us = max(0, skip_us - shift_us)

    try:
        with h5py.File(out_path, 'w') as f:
            grp = f.create_group('evk4_hd')
            print(f'[master] skip={skip_us} us, t_end={t_end_us} us', flush=True)
            n_left = stream_raw_to_hdf5(args.master, grp, 'left/events',
                                        time_skip_us=skip_us, time_end_us=t_end_us)
            print(f'[master] {n_left:,} events written', flush=True)
            print(f'[slave]  raw_seek={slave_seek_us} us, shift={shift_us:+d} us, '
                  f't_end={t_end_us} us', flush=True)
            n_right = stream_raw_to_hdf5(args.slave, grp, 'right/events',
                                         time_skip_us=slave_seek_us,
                                         time_shift_us=shift_us,
                                         time_end_us=t_end_us)
            print(f'[slave]  {n_right:,} events written', flush=True)
        print(f'[done] {out_path}', flush=True)
    except BaseException as e:
        tb = traceback.format_exc()
        print('[ERROR] ' + tb, flush=True)
        with open(crash_log, 'w') as fh:
            fh.write(tb)
        print(f'[ERROR] traceback at {crash_log}', flush=True)
        raise


if __name__ == '__main__':
    main()
