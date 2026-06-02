"""
verify_rectification.py  —  Sanity-check the stereo rectification maps produced
by StereoCalibration.

Usage (no real images needed):
    python verify_rectification.py path/to/camchain.yaml

Usage (with test images):
    python verify_rectification.py path/to/camchain.yaml \
        --left  path/to/left.png \
        --right path/to/right.png

Checks performed
----------------
1. Map shape & dtype           — both maps are (H, W, 2) float32.
2. Map coverage                — no NaN / Inf values, coordinates stay
                                 within sensor bounds.
3. Epipolar alignment          — for a grid of point correspondences, the
                                 y-difference after rectification should be
                                 near zero (|Δy| < 1 px on average).
4. Horizontal-only disparity   — same point grid: x_left < x_right for a
                                 scene point at finite depth.
5. Q matrix sanity             — Q[3,2] encodes 1/baseline; Q[2,3] is fx.
6. Valid ROI sanity            — ROI lies inside image bounds and is non-empty.
7. Visual check (optional)     — if images are supplied, saves side-by-side
                                 rectified images with horizontal epipolar lines
                                 to  rectification_check.png.
"""

import argparse
import sys
import numpy as np
import cv2

# Allow running from anywhere inside the workspace
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.Custom.calibration import StereoCalibration


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check(name: str, ok: bool, detail: str = ''):
    tag = 'PASS' if ok else 'FAIL'
    msg = f'  [{tag}] {name}'
    if detail:
        msg += f'  —  {detail}'
    print(msg)
    return ok


def _sample_grid(calib: StereoCalibration, n: int = 20):
    """
    Return (n*n, 2) arrays of (x, y) pixel centres on a regular grid
    covering the sensor, and their rectified counterparts for left/right.
    """
    w, h = calib.image_size
    xs = np.linspace(0, w - 1, n, dtype=np.float32)
    ys = np.linspace(0, h - 1, n, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)          # (n, n)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (n*n, 2)

    def lookup(rmap, pts):
        xi = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
        yi = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
        xr = rmap[yi, xi, 0]
        yr = rmap[yi, xi, 1]
        return np.stack([xr, yr], axis=1)

    left_rect  = lookup(calib.left_map,  pts)
    right_rect = lookup(calib.right_map, pts)
    return pts, left_rect, right_rect


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def check_shape_dtype(calib: StereoCalibration) -> bool:
    w, h = calib.image_size
    ok = True
    for name, rmap in [('left_map', calib.left_map), ('right_map', calib.right_map)]:
        expected = (h, w, 2)
        shape_ok = rmap.shape == expected
        dtype_ok = rmap.dtype == np.float32
        ok &= _check(f'{name} shape', shape_ok,
                     f'got {rmap.shape}, expected {expected}')
        ok &= _check(f'{name} dtype', dtype_ok,
                     f'got {rmap.dtype}, expected float32')
    return ok


def check_coverage(calib: StereoCalibration) -> bool:
    w, h = calib.image_size
    ok = True
    for name, rmap in [('left_map', calib.left_map), ('right_map', calib.right_map)]:
        nan_inf = not np.isfinite(rmap).all()
        ok &= _check(f'{name} no NaN/Inf', not nan_inf,
                     f'{np.sum(~np.isfinite(rmap))} bad values' if nan_inf else '')
        # Most pixels should map to inside the sensor (some border wrap is OK)
        in_bounds = (
            (rmap[..., 0] >= 0) & (rmap[..., 0] < w) &
            (rmap[..., 1] >= 0) & (rmap[..., 1] < h)
        )
        pct = 100.0 * in_bounds.sum() / in_bounds.size
        ok &= _check(f'{name} in-bounds coverage', pct > 90.0,
                     f'{pct:.1f}% pixels map inside sensor')
    return ok


def check_epipolar(calib: StereoCalibration) -> bool:
    _, left_rect, right_rect = _sample_grid(calib)
    # After rectification epipolar lines are horizontal → y_left ≈ y_right
    dy = np.abs(left_rect[:, 1] - right_rect[:, 1])
    mean_dy = float(np.nanmean(dy))
    max_dy  = float(np.nanmax(dy))
    ok = mean_dy < 1.0
    _check('Epipolar alignment (mean |Δy|)', ok,
           f'mean={mean_dy:.3f} px, max={max_dy:.3f} px')
    return ok


def check_disparity_direction(calib: StereoCalibration) -> bool:
    """
    For a standard left-right stereo rig, rectified x_left > x_right
    (left image has larger x for the same scene point).
    We check the sign is consistent for interior grid points.
    """
    _, left_rect, right_rect = _sample_grid(calib)
    dx = left_rect[:, 0] - right_rect[:, 0]
    # Ignore points that mapped out of bounds (dx near ±sensor_width)
    w, _ = calib.image_size
    finite_mask = (np.abs(dx) < w * 0.9)
    if finite_mask.sum() == 0:
        return _check('Disparity direction', False, 'no valid sample points')
    positive = dx[finite_mask] > 0
    pct = 100.0 * positive.sum() / finite_mask.sum()
    ok = pct > 80.0   # majority should be positive disparity
    _check('Disparity direction (x_left > x_right)', ok,
           f'{pct:.1f}% of samples have positive disparity')
    return ok


def check_q_matrix(calib: StereoCalibration) -> bool:
    Q = calib.Q
    ok = True
    # Q[3,2] = -1/baseline  (negative in OpenCV convention)
    baseline_q = -1.0 / Q[3, 2] if abs(Q[3, 2]) > 1e-9 else np.inf
    # Q[2,3] = fx (focal length in rectified image)
    fx_q = Q[2, 3]
    ok &= _check('Q baseline finite & positive', np.isfinite(baseline_q) and baseline_q > 0,
                 f'baseline = {baseline_q:.5f} m')
    ok &= _check('Q focal length positive', fx_q > 0,
                 f'fx = {fx_q:.2f} px')
    # Cross-check: P_rect[0,0] should equal fx_q
    fx_p = calib.P_rect[0, 0]
    ok &= _check('Q fx matches P_rect[0,0]', abs(fx_p - fx_q) < 1.0,
                 f'P_rect fx={fx_p:.2f}, Q fx={fx_q:.2f}')
    return ok


def check_valid_roi(calib: StereoCalibration) -> bool:
    w, h = calib.image_size
    x, y, rw, rh = calib.valid_roi
    ok = True
    ok &= _check('Valid ROI non-empty',   rw > 0 and rh > 0,
                 f'ROI = ({x},{y},{rw},{rh})')
    ok &= _check('Valid ROI inside image', x >= 0 and y >= 0
                 and x + rw <= w and y + rh <= h,
                 f'image {w}x{h}, ROI ({x},{y},{rw},{rh})')
    area_pct = 100.0 * rw * rh / (w * h)
    ok &= _check('Valid ROI covers > 20% of sensor', area_pct > 20.0,
                 f'{area_pct:.1f}%')
    return ok


# ---------------------------------------------------------------------------
# optional visual check
# ---------------------------------------------------------------------------

def visual_check(calib: StereoCalibration, left_path: str, right_path: str,
                 out_path: str = 'rectification_check.png'):
    left_img  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if left_img is None or right_img is None:
        print(f'  [WARN] Could not read images: {left_path}, {right_path}')
        return

    left_rect  = calib.rectify_image(left_img,  'left')
    right_rect = calib.rectify_image(right_img, 'right')

    # Convert to BGR for coloured epipolar lines
    l_bgr = cv2.cvtColor(left_rect,  cv2.COLOR_GRAY2BGR)
    r_bgr = cv2.cvtColor(right_rect, cv2.COLOR_GRAY2BGR)

    h, w = l_bgr.shape[:2]
    line_color = (0, 255, 0)
    n_lines = 20
    for y in np.linspace(0, h - 1, n_lines, dtype=int):
        cv2.line(l_bgr, (0, y), (w - 1, y), line_color, 1)
        cv2.line(r_bgr, (0, y), (w - 1, y), line_color, 1)

    # Draw valid ROI rectangle
    rx, ry, rw_roi, rh_roi = calib.valid_roi
    roi_color = (0, 100, 255)
    cv2.rectangle(l_bgr, (rx, ry), (rx + rw_roi, ry + rh_roi), roi_color, 2)
    cv2.rectangle(r_bgr, (rx, ry), (rx + rw_roi, ry + rh_roi), roi_color, 2)

    combined = np.hstack([l_bgr, r_bgr])
    cv2.imwrite(out_path, combined)
    print(f'  [INFO] Visual check saved to: {out_path}')
    print( '         Green lines = epipolar lines (should touch same features)')
    print( '         Orange box  = valid stereo ROI')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Verify stereo rectification maps.')
    parser.add_argument('yaml',  help='Path to Kalibr camchain YAML file')
    parser.add_argument('--left',  default=None, help='Left camera test image')
    parser.add_argument('--right', default=None, help='Right camera test image')
    parser.add_argument('--out',   default='rectification_check.png',
                        help='Output path for visual check image')
    args = parser.parse_args()

    print(f'\nLoading calibration from: {args.yaml}')
    calib = StereoCalibration(args.yaml)

    print(f'\n  Image size  : {calib.image_size[0]} x {calib.image_size[1]}  (W x H)')
    print(f'  Baseline    : {calib.baseline:.6f} m')
    print(f'  Focal length: {calib.focal_length_x:.2f} px')

    print('\n── Numerical checks ──────────────────────────────────────────────')
    results = [
        check_shape_dtype(calib),
        check_coverage(calib),
        check_epipolar(calib),
        check_disparity_direction(calib),
        check_q_matrix(calib),
        check_valid_roi(calib),
    ]

    if args.left and args.right:
        print('\n── Visual check ──────────────────────────────────────────────────')
        visual_check(calib, args.left, args.right, args.out)

    passed = sum(results)
    total  = len(results)
    print(f'\n══ Result: {passed}/{total} checks passed '
          + ('✓' if passed == total else '✗'))

    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
