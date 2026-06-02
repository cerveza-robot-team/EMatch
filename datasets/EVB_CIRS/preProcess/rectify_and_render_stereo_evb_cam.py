"""
Stereo Event Camera Rectification and Rendering
================================================
Loads events from CIRS_EVB HDF5 file, computes stereo rectification maps
from the camchain YAML via StereoCalibration, accumulates events into frames,
rectifies them, crops to the valid stereo ROI, and writes a side-by-side
stereo video.

Usage:
    python rectify_and_render_stereo_evb_cam.py

Outputs (saved next to this script):
    rectify_maps_left.npz   -- map1/map2 for left camera (float32)
    rectify_maps_right.npz  -- map1/map2 for right camera (float32)
    stereo_rectified.mp4    -- side-by-side rectified video (cropped to valid ROI)
    stereo_overlay.mp4      -- per-event overlay video (left=red, right=green)
"""

import os
import sys
import numpy as np
import cv2
import h5py

EMATCH_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_ROOT = os.path.join(EMATCH_DIR, "data", "EVB_CIRS")
HDF5_DIR = os.path.join(DATASET_ROOT, "hdf5")
CONFIG_PATH = os.path.join(DATASET_ROOT, "calibration")

# Make EMatch package importable
sys.path.insert(0, EMATCH_DIR)
from datasets.EVB_CIRS.calibration import StereoCalibration  # type: ignore  # noqa: E402

# ─── Paths ───────────────────────────────────────────────────────────────────

HDF5_PATH        = os.path.join(HDF5_DIR, "1_0m_out.hdf5")
CONFIG_YAML_PATH        = os.path.join(CONFIG_PATH, "camchain-cirs_evb.yaml")
OUT_MAP_L        = os.path.join(CONFIG_PATH, "rectify_maps_left.npz")
OUT_MAP_R        = os.path.join(CONFIG_PATH, "rectify_maps_right.npz")
OUT_VIDEO        = os.path.join(DATASET_ROOT, "render", "stereo_rectified_1m_outdoor_64.mp4")
OUT_VIDEO_OVERLAY = os.path.join(DATASET_ROOT, "render", "stereo_overlay_1m_outdoor_64.mp4")

# ─── Rendering parameters ────────────────────────────────────────────────────

FRAME_DURATION_US = 33_000   # accumulation window per frame (~30 fps)
FPS               = 30
POLARITY_COLORS   = {
     1.0: (255, 255, 255),   # positive → white
    -1.0: (100, 100, 255),   # negative → blue-ish
}


# ─── 1. Accumulate events into a BGR frame ───────────────────────────────────

def events_to_frame(events_chunk, W: int, H: int) -> np.ndarray:
    """events_chunk: Nx4 array [x, y, pol, ts]"""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    if len(events_chunk) == 0:
        return frame
    xs = events_chunk[:, 0].astype(np.int32)
    ys = events_chunk[:, 1].astype(np.int32)
    ps = events_chunk[:, 3]

    # Draw negative first so positive events overwrite on overlap
    for pol, color in [(-1.0, POLARITY_COLORS[-1.0]), (1.0, POLARITY_COLORS[1.0])]:
        mask = ps == pol
        frame[ys[mask], xs[mask]] = color

    return frame


def events_to_mask(events_chunk, W: int, H: int) -> np.ndarray:
    """Return a single-channel uint8 mask: 255 wherever any event fired."""
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(events_chunk) == 0:
        return mask
    xs = events_chunk[:, 0].astype(np.int32)
    ys = events_chunk[:, 1].astype(np.int32)
    mask[ys, xs] = 255
    return mask


def load_events(path, dataset="CD/events"):
    """Load events from HDF5, always returning a plain float64 array [x, y, pol, ts].

    Handles both:
        - structured array  with named fields (x, y, p, t)  — individual cam files
        - plain 2D array    columns [x, y, pol, ts]          — combined stereo file
    """
    with h5py.File(path, "r") as f:
        raw = f[dataset][:]
    if raw.dtype.names:                          # structured / named fields
        x  = raw["x"].astype(np.float64)
        y  = raw["y"].astype(np.float64)
        ts = raw["t"].astype(np.float64)
        p  = raw["p"].astype(np.float64) * 2 - 1  # {0,1} → {-1,+1}
        return np.column_stack([x, y, p, ts])    # [x, y, pol, ts]
    else:                                        # plain 2D array — already [x, y, pol, ts]
        return raw.astype(np.float64)


# ─── 2. Main ─────────────────────────────────────────────────────────────────

def main():
    # ── Load stereo calibration ──────────────────────────────────────────────
    print("Loading stereo calibration …")
    calib = StereoCalibration(CONFIG_YAML_PATH, expected_disparity=64)

    W, H = calib.image_size

    print(f"\n=== Camera Parameters Summary ===")
    print(f"Sensor size       : {W} x {H}")
    print(f"Left  K :\n{np.round(calib.K_left,  2)}")
    print(f"Left  D : {np.round(calib.D_left,  4)}")
    print(f"Right K :\n{np.round(calib.K_right, 2)}")
    print(f"Right D : {np.round(calib.D_right,  4)}")
    print(f"Rectified focal length : {calib.focal_length_x:.2f} px")
    print(f"Baseline               : {calib.baseline_t * 100:.2f} cm")

    # ── Use StereoCalibration rectification maps and valid ROI ───────────────
    # calib already computed full stereo rectification maps (undistort + rectify)
    # and the intersected valid ROI via cv2.stereoRectify(alpha=0).
    crop_x0, crop_y0, crop_w, crop_h = calib.valid_roi
    print(f"  Valid crop region  : {crop_w} x {crop_h}  (offset {crop_x0}, {crop_y0})")

    # ── Save rectification maps ──────────────────────────────────────────────
    np.savez(OUT_MAP_L, map_x=calib.left_map[..., 0],  map_y=calib.left_map[..., 1])
    np.savez(OUT_MAP_R, map_x=calib.right_map[..., 0], map_y=calib.right_map[..., 1])
    print(f"\n  Saved left  maps → {OUT_MAP_L}")
    print(f"  Saved right maps → {OUT_MAP_R}")

    # ── Load events ──────────────────────────────────────────────────────────
    print(f"\nLoading events from {HDF5_PATH} …")
    ev_left  = load_events(HDF5_PATH, dataset="evk4_hd/left/events")
    ev_right = load_events(HDF5_PATH, dataset="evk4_hd/right/events")

    print(f"  Left  events: {len(ev_left)},   sample: {ev_left[0]}")
    print(f"  Right events: {len(ev_right)},  sample: {ev_right[0]}")

    # Sort by timestamp (columns: x, y, pol, ts)
    ev_left  = ev_left [ev_left [:, 2].argsort()]
    ev_right = ev_right[ev_right[:, 2].argsort()]

    # ── Time synchronisation ─────────────────────────────────────────────────
    t0_l, t1_l = ev_left [0, 2], ev_left [-1, 2]
    t0_r, t1_r = ev_right[0, 2], ev_right[-1, 2]
    t0 = max(t0_l, t0_r)   # latest start  (both cameras active)
    t1 = min(t1_l, t1_r)   # earliest end  (both cameras active)
    offset_us = t0_l - t0_r

    print(f"  Left  ts: {t0_l:.0f} … {t1_l:.0f} µs  ({(t1_l-t0_l)/1e6:.2f} s)")
    print(f"  Right ts: {t0_r:.0f} … {t1_r:.0f} µs  ({(t1_r-t0_r)/1e6:.2f} s)")
    print(f"  Start offset (left − right): {offset_us:.0f} µs")
    print(f"  Sync window:  {t0:.0f} … {t1:.0f} µs  ({(t1-t0)/1e6:.2f} s)")

    duration_us = t1 - t0
    n_frames = int(np.ceil(duration_us / FRAME_DURATION_US))
    print(f"  → {n_frames} frames @ {FRAME_DURATION_US/1000:.1f} ms/frame")

    ts_left  = ev_left [:, 2]
    ts_right = ev_right[:, 2]

    idx_l = int(np.searchsorted(ts_left,  t0, side="left"))
    idx_r = int(np.searchsorted(ts_right, t0, side="left"))

    # ── Video writers — sized to the valid crop after rotation ───────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer         = cv2.VideoWriter(OUT_VIDEO,         fourcc, FPS, (2 * crop_w, crop_h))
    writer_overlay = cv2.VideoWriter(OUT_VIDEO_OVERLAY, fourcc, FPS, (crop_w, crop_h))

    print("Rendering and rectifying frames …")

    for i in range(n_frames):
        t_start = t0 + i * FRAME_DURATION_US
        t_end   = t_start + FRAME_DURATION_US

        end_l = int(np.searchsorted(ts_left,  t_end, side="left"))
        end_r = int(np.searchsorted(ts_right, t_end, side="left"))

        chunk_l = ev_left [idx_l:end_l]
        chunk_r = ev_right[idx_r:end_r]
        idx_l, idx_r = end_l, end_r

        # ── Side-by-side video ───────────────────────────────────────────────
        frame_l = events_to_frame(chunk_l, W, H)
        frame_r = events_to_frame(chunk_r, W, H)

        # Rectify using StereoCalibration then crop to valid ROI
        rect_l = calib.rectify_image(frame_l, 'left') [crop_y0:crop_y0 + crop_h, crop_x0:crop_x0 + crop_w]
        rect_r = calib.rectify_image(frame_r, 'right')[crop_y0:crop_y0 + crop_h, crop_x0:crop_x0 + crop_w]

        # Draw horizontal epipolar lines every 60 px to verify alignment
        for y in range(0, crop_h, 60):
            cv2.line(rect_l, (0, y), (crop_w - 1, y), (0, 200, 0), 1)
            cv2.line(rect_r, (0, y), (crop_w - 1, y), (0, 200, 0), 1)

        label    = f"frame {i+1}/{n_frames}"
        ts_label = f"ts {t_start/1e3:.1f} - {t_end/1e3:.1f} ms  (dt={FRAME_DURATION_US/1e3:.1f} ms)"
        cv2.putText(rect_l, f"LEFT  {label}",  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)
        cv2.putText(rect_l, ts_label,           (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 3)
        cv2.putText(rect_r, f"RIGHT {label}",  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)
        cv2.putText(rect_r, ts_label,           (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 3)

        writer.write(np.hstack([rect_l, rect_r]))

        # ── Overlay video: per-event rectification via StereoCalibration ────
        rect_ev_l = calib.rectify_events(chunk_l, 'left')
        rect_ev_r = calib.rectify_events(chunk_r, 'right')

        mask_l = events_to_mask(rect_ev_l, crop_w, crop_h)
        mask_r = events_to_mask(rect_ev_r, crop_w, crop_h)

        overlay = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        overlay[:, :, 2] = mask_l   # R channel = left camera
        overlay[:, :, 1] = mask_r   # G channel = right camera
        cv2.putText(overlay, f"LEFT (red)   RIGHT (green)   OVERLAP (yellow)   {label}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(overlay, ts_label, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
        for y in range(0, crop_h, 60):
            cv2.line(overlay, (0, y), (crop_w - 1, y), (0, 80, 0), 1)
        writer_overlay.write(overlay)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_frames} frames …")

    writer.release()
    writer_overlay.release()
    print(f"\nDone.")
    print(f"  Side-by-side → {OUT_VIDEO}")
    print(f"  Overlay      → {OUT_VIDEO_OVERLAY}")

    print("\n=== Rectification Summary ===")
    print(f"Rectified focal length : {calib.focal_length_x:.2f} px")
    print(f"Baseline               : {calib.baseline_t * 100:.2f} cm")
    print(f"Crop region            : {crop_w} x {crop_h}  (offset {crop_x0}, {crop_y0})")
    print(f"Output video size      : {2*crop_w} x {crop_h}  (side-by-side)")


if __name__ == "__main__":
    main()
