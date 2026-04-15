"""
Stereo calibration loader for custom event cameras.

Supports Kalibr camchain YAML format (standard output from event camera calibration):
    https://github.com/ethz-asl/kalibr

Expected YAML structure (two cameras, cam0 = left, cam1 = right):

    cam0:
      camera_model: pinhole
      intrinsics: [fx, fy, cx, cy]
      distortion_model: radtan        # or 'radial' or 'equidist'
      distortion_coeffs: [k1, k2, p1, p2]   # radtan: 4 coeffs
                                             # radial: [k1, k2, k3]  (no tangential)
                                             # equidist: [k1, k2, k3, k4]
      resolution: [width, height]

    cam1:
      camera_model: pinhole
      intrinsics: [fx, fy, cx, cy]
      distortion_model: radtan
      distortion_coeffs: [k1, k2, p1, p2]
      resolution: [width, height]
      T_cn_cnm1:            # 4x4 transform: cam1 expressed in cam0 frame
        - [r00, r01, r02, tx]
        - [r10, r11, r12, ty]
        - [r20, r21, r22, tz]
        - [0,   0,   0,   1 ]

After loading, `StereoCalibration` provides:
    left_map  : (H, W, 2) float32  — cv2.remap map for left camera
    right_map : (H, W, 2) float32  — cv2.remap map for right camera
    R_rect    : (3, 3)              — rectification rotation (left camera)
    P_rect    : (3, 4)              — rectified projection matrix (left camera)
    Q         : (4, 4)              — disparity-to-depth mapping matrix
"""

import yaml
import numpy as np
import cv2


class StereoCalibration:
    def __init__(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            calib = yaml.safe_load(f)

        cam0 = calib['cam0']
        cam1 = calib['cam1']

        # --- intrinsics ---
        fx0, fy0, cx0, cy0 = cam0['intrinsics']
        fx1, fy1, cx1, cy1 = cam1['intrinsics']

        self.K_left  = np.array([[fx0, 0, cx0],
                                  [0, fy0, cy0],
                                  [0,   0,   1]], dtype=np.float64)
        self.K_right = np.array([[fx1, 0, cx1],
                                  [0, fy1, cy1],
                                  [0,   0,   1]], dtype=np.float64)

        # --- distortion ---
        self.dist_model = cam0.get('distortion_model', 'radtan')  # radtan, radial, or equidist
        self.D_left  = self._parse_distortion(cam0['distortion_coeffs'], self.dist_model)
        self.D_right = self._parse_distortion(cam1['distortion_coeffs'], self.dist_model)

        # --- extrinsics: T_cam0_cam1 (rigid body transform) ---
        # --- From cam0 to cam1, i.e. cam1 expressed in cam0 frame ---
        T = np.array(cam1['T_cn_cnm1'], dtype=np.float64)  # 4x4
        R = T[:3, :3]   # rotation from cam0 → cam1
        t = T[:3, 3]    # translation of cam1 origin in cam0 frame

        # --- image size ---
        w, h = cam0['resolution']
        self.image_size = (w, h)

        # --- compute stereo rectification ---
        self.left_map, self.right_map, self.R_rect, self.R_rect_right, self.P_rect, self.Q, \
            self.valid_roi = self._compute_rectification_maps(R, t, w, h)
        self.R_rect_left = self.R_rect   # alias for clarity

        self.baseline = float(np.abs(self.Q[3, 2]))   # 1/baseline (from Q matrix)
        self.focal_length_x = float(self.Q[2, 3])     # focal length after rectification

        # Suggest a crop_size that fits inside the valid rectified region,
        # is divisible by 32, and preserves the sensor aspect ratio as closely as possible.
        self.suggested_crop_size = self._suggest_crop_size(self.valid_roi)
        x, y, rw, rh = self.valid_roi
        print(f'  Valid rectified ROI : {rw} x {rh}  (offset {x}, {y})')
        print(f'  Suggested crop_size : {self.suggested_crop_size}  '
              f'[H, W]  (divisible by 32, fits inside valid ROI)')

    @staticmethod
    def _parse_distortion(coeffs: list, model: str) -> np.ndarray:
        """
        Convert distortion coefficients to the format OpenCV expects.

        radtan  : [k1, k2, p1, p2]          → pass as-is
        radial  : [k1, k2, k3]              → [k1, k2, 0, 0, k3]  (no tangential)
        equidist: [k1, k2, k3, k4]          → pass as-is (used with fisheye API)
        """
        d = np.array(coeffs, dtype=np.float64)
        if model == 'radial':
            # OpenCV convention: (k1, k2, p1, p2, k3)
            k1, k2, k3 = d[0], d[1], d[2] if len(d) > 2 else 0.0
            d = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)
        return d

    def _compute_rectification_maps(self, R, t, w, h):
        # Full stereo rectification: compute R1, R2, P1, P2, Q so that
        # epipolar lines are horizontal in both rectified images.
        # alpha=0 → crop to the largest rectangle of fully valid pixels.
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K_left,  self.D_left,
            self.K_right, self.D_right,
            (w, h), R, t, alpha=0
        )

        map_l_x, map_l_y = cv2.initUndistortRectifyMap(
            self.K_left,  self.D_left,  R1, P1, (w, h), cv2.CV_32FC1)
        map_r_x, map_r_y = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, R2, P2, (w, h), cv2.CV_32FC1)

        # Compute valid ROI directly from the remap maps.
        # cv2.stereoRectify's roi1/roi2 are unreliable with large rotations
        # (e.g. toe-in rigs) and often return the full-frame rectangle even
        # when corners have been rotated out of bounds.  Checking the maps
        # directly — keeping only rows/cols where EVERY pixel maps back inside
        # the original sensor — gives the true tight crop.
        valid_l = ((map_l_x >= 0) & (map_l_x <= w - 1) &
                   (map_l_y >= 0) & (map_l_y <= h - 1))
        valid_r = ((map_r_x >= 0) & (map_r_x <= w - 1) &
                   (map_r_y >= 0) & (map_r_y <= h - 1))
        valid   = valid_l & valid_r

        row_mask = np.all(valid, axis=1)
        col_mask = np.all(valid, axis=0)
        rows = np.where(row_mask)[0]
        cols = np.where(col_mask)[0]
        roi_x  = int(cols[0]);  roi_y  = int(rows[0])
        roi_w  = int(cols[-1] - cols[0] + 1)
        roi_h  = int(rows[-1] - rows[0] + 1)
        valid_roi = (roi_x, roi_y, roi_w, roi_h)

        # Stack into (H, W, 2) — same convention as MVSEC calibration
        left_map  = np.stack([map_l_x, map_l_y], axis=2)  # (H, W, 2)
        right_map = np.stack([map_r_x, map_r_y], axis=2)  # (H, W, 2)

        return left_map, right_map, R1, R2, P1, Q, valid_roi

    @staticmethod
    def _suggest_crop_size(valid_roi):
        """
        Given the valid rectified ROI (x, y, w, h), return [H, W] rounded down
        to the nearest multiple of 32 — the minimum stride of the backbone.
        """
        _, _, rw, rh = valid_roi
        h32 = (rh // 32) * 32
        w32 = (rw // 32) * 32
        return [h32, w32]

    def rectify_image(self, image: np.ndarray, side: str) -> np.ndarray:
        """
        Rectify a grayscale or colour image using the precomputed stereo maps.

        :param image: (H, W) or (H, W, C) array
        :param side:  'left' or 'right'
        :return:      rectified image, same dtype and shape as input
        """
        rmap = self.left_map if side == 'left' else self.right_map
        return cv2.remap(image, rmap[..., 0], rmap[..., 1], cv2.INTER_LINEAR)

    def rectify_events(self, events: np.ndarray, side: str) -> np.ndarray:
        """
        Remap event (x, y) coordinates using the precomputed rectification maps,
        crop to the valid stereo ROI, and shift coordinates so (0, 0) is the ROI origin.

        :param events: [N, 4] array (x, y, t, p)
        :param side:   'left' or 'right'
        :return:       [M, 4] array with rectified (x, y) in ROI-local coordinates
        """
        rmap = self.left_map if side == 'left' else self.right_map
        w, h = self.image_size
        roi_x, roi_y, roi_w, roi_h = self.valid_roi

        xs = events[:, 0].astype(np.int32)
        ys = events[:, 1].astype(np.int32)

        # Clamp to valid sensor range before lookup
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

        # Look up rectified coordinates
        x_rect = np.round(rmap[ys, xs, 0]).astype(np.int32)
        y_rect = np.round(rmap[ys, xs, 1]).astype(np.int32)

        # Keep only events inside the valid stereo ROI
        valid = (
            (x_rect >= roi_x) & (x_rect < roi_x + roi_w) &
            (y_rect >= roi_y) & (y_rect < roi_y + roi_h)
        )

        rectified = events[valid].copy()
        # Shift so ROI top-left maps to (0, 0)
        rectified[:, 0] = x_rect[valid] - roi_x
        rectified[:, 1] = y_rect[valid] - roi_y

        return rectified
