"""
Generic dataset loader for custom stereo event camera data in MVSEC-like HDF5 format.

Expected HDF5 structure:
    davis/left/events   - [N, 4] array of (x, y, t, p)
    davis/right/events  - [N, 4] array of (x, y, t, p)

Where:
    x, y   : pixel coordinates
    t      : timestamps in seconds
    p      : polarity in {-1, +1} or {0, 1} (both are handled)

If your data uses a different group path, set 'event_path_left' and 'event_path_right'
in the config YAML to override the defaults.
"""

import os
import numpy as np
import torch
import h5py
import cv2
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.MVSEC.utils.EventToVoxel import events_to_voxel
from datasets.EVB_CIRS.calibration import StereoCalibration

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class CustomHDF5Reader:
    """
    Reads events from a single HDF5 file with MVSEC-like structure.

    The default group path is 'evk4_hd/<location>/events' which stores a [N,4]
    array of (x, y, t, p).  Override with event_path_left / event_path_right
    in the config if your file uses a different path.
    """

    def __init__(self, hdf5_path: str, location: str = 'left', event_path: str = None):
        """
        :param hdf5_path: path to the .hdf5 file
        :param location:  'left' or 'right'
        :param event_path: optional override for the HDF5 dataset path,
                           e.g. 'events/data' — defaults to 'evk4_hd/<location>/events'
        """
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f'HDF5 file not found: {hdf5_path}')

        self.f = h5py.File(hdf5_path, 'r')
        path = event_path if event_path else f'evk4_hd/{location}/events'
        if path not in self.f:
            raise KeyError(
                f"HDF5 path '{path}' not found in {hdf5_path}. "
                f"Available top-level keys: {list(self.f.keys())}. "
                f"Set 'event_path_left' / 'event_path_right' in your config to override."
            )
        self.event_data = self.f[path][:]   # [N, 4]: x, y, t, p  (load into memory)
        raw_ts = self.event_data[:, 2]
        # Detect timestamp unit from the recording *range* (delta), not the absolute
        # value.  Using absolute value misclassifies Unix timestamps in seconds
        # (~1.7e9) as nanoseconds.  A 0.1 s recording has:
        #   ns  → delta ~1e8   µs → delta ~1e5   ms → delta ~100   s → delta ~0.1
        delta = float(raw_ts[-1] - raw_ts[0]) if len(raw_ts) > 1 else 0.0
        if delta > 1e8:            # nanoseconds
            self._ts_scale = 1e-9
        elif delta > 1e5:          # microseconds
            self._ts_scale = 1e-6
        elif delta > 100:          # milliseconds
            self._ts_scale = 1e-3
        else:                      # already seconds
            self._ts_scale = 1.0
        self.event_ts = raw_ts * self._ts_scale  # timestamps in seconds

    def find_ts_index(self, t: float) -> int:
        """Binary search for the index of timestamp t in event_ts."""
        l, r = 0, len(self.event_ts) - 1
        while l <= r:
            mid = (l + r) // 2
            if self.event_ts[mid] < t:
                l = mid + 1
            else:
                r = mid - 1
        return l

    def get_events(self, start: int, end: int) -> np.ndarray:
        """
        Return events[start:end] as [M, 4] array (x, y, t, p).
        Polarity is normalized to {-1, +1}.
        """
        events = self.event_data[start:end].astype(np.float64)
        if events.shape[0] > 0:
            events[:, 2] *= self._ts_scale  # convert timestamps to seconds
            # Normalize polarity: if stored as {0,1} convert to {-1,+1}
            unique_pols = np.unique(events[:, 3])
            if np.all(unique_pols >= 0):  # stored as {0, 1}
                events[:, 3] = events[:, 3] * 2.0 - 1.0
        return events

    def get_timestamps(self) -> np.ndarray:
        """Return all event timestamps (seconds)."""
        return self.event_ts[:]

    def __len__(self) -> int:
        return len(self.event_data)



class CustomSequence_ematch(Dataset):
    """
    Generic inference-only dataset for stereo event cameras.

    Config keys (all required unless marked optional):
        root_path       : path to the HDF5 file (e.g. '/data/my_recording.hdf5')
        dt              : event time window in milliseconds (e.g. 100)
        voxel_bins      : number of temporal bins (e.g. 15)
        crop_size       : [H, W] — voxels are resized to this before inference.
                          Must be divisible by 32.
        sensor_height   : native sensor height (pixels)
        sensor_width    : native sensor width  (pixels)
        event_path_left : (optional) HDF5 path to left events, default 'davis/left/events'
        event_path_right: (optional) HDF5 path to right events, default 'davis/right/events'
        stride          : (optional) step between consecutive time windows in ms, default = dt
    """

    def __init__(self, cfgs, name: str = 'custom'):
        self.name = name
        hdf5_path = cfgs['root_path']

        self.dt = cfgs['dt']                    # milliseconds
        self.voxel_bins = cfgs['voxel_bins']
        self.crop_size = cfgs['crop_size']      # [H, W]
        self.sensor_h = cfgs['sensor_height']
        self.sensor_w = cfgs['sensor_width']
        self.stride = cfgs.get('stride', self.dt)  # ms between windows
        self.expected_disparity = cfgs.get('expected_disparity_at_1m', None)  # for extrinsics sanity check
        self.alpha = cfgs.get('alpha', 0.5)  # for disparity shift correction based on expected disparity at 1m

        crop_h, crop_w = self.crop_size
        assert crop_h % 32 == 0 and crop_w % 32 == 0, (
            f'crop_size {self.crop_size} must be divisible by 32 '
            f'to satisfy the backbone stride requirements.'
        )

        path_left  = cfgs.get('event_path_left',  None)
        path_right = cfgs.get('event_path_right', None)
        self.reader_left  = CustomHDF5Reader(hdf5_path, 'left',  path_left)
        self.reader_right = CustomHDF5Reader(hdf5_path, 'right', path_right)

        # Optional stereo calibration for event rectification
        calib_path = cfgs.get('calib_path', None)
        if calib_path:
            self.calib = StereoCalibration(calib_path, alpha=self.alpha, expected_disparity=self.expected_disparity)
            print(f'Loaded stereo calibration from {calib_path}')
        else:
            self.calib = None
            print('WARNING: no calib_path set — events will NOT be rectified. '
                  'Disparity results may be incorrect.')

        # Build list of (T_start, T_end) windows in seconds
        self._windows = self._build_windows()

    def _build_windows(self):
        """
        Divide the recording into fixed-size time windows of dt ms,
        stepping by stride ms.  Mirrors MVSEC: T_start = T_end - dt * 0.001.
        All timestamps are kept in seconds throughout.
        """
        dt_sec     = self.dt     * 0.001   # ms → s  (same factor as MVSEC)
        stride_sec = self.stride * 0.001

        ts_left  = self.reader_left.get_timestamps()   # seconds
        ts_right = self.reader_right.get_timestamps()  # seconds

        t_start_global = max(ts_left[0],  ts_right[0])
        t_end_global   = min(ts_left[-1], ts_right[-1])

        total_time = t_end_global - t_start_global
        if total_time <= 0:
            raise RuntimeError(
                'No overlapping events found between left and right cameras. '
                f'Left camera time range: [{ts_left[0]:.3f} s, {ts_left[-1]:.3f} s], '
                f'Right camera time range: [{ts_right[0]:.3f} s, {ts_right[-1]:.3f} s].'
            )
        print(f'Global time range: [{t_start_global:.3f} s, {t_end_global:.3f} s]')
        print(f'Total recording duration: {total_time:.3f} s')
        print(f'Building time windows with dt={self.dt} ms and stride={self.stride} ms...')

        windows = []
        t_end = t_start_global + dt_sec   # first window ends here
        while t_end <= t_end_global:
            windows.append((t_end - dt_sec, t_end))  # (T_start, T_end) in seconds
            t_end += stride_sec

        if len(windows) == 0:
            raise RuntimeError(
                'No valid time windows found. Check that left and right event streams overlap '
                f'and that dt={self.dt}ms is appropriate for your data.'
            )
        print(f'Built {len(windows)} time windows.')
        return windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int):
        t_start, t_end = self._windows[index]  # already in seconds

        voxel_0 = self._load_voxel(self.reader_left,  t_start, t_end, side='left')
        voxel_1 = self._load_voxel(self.reader_right, t_start, t_end, side='right')

        # Center-crop to crop_size — mirrors MVSEC's non-augmented path
        crop_h, crop_w = self.crop_size
        h, w = voxel_0.shape[1], voxel_0.shape[2]
        if h != crop_h or w != crop_w:
            assert crop_h <= h and crop_w <= w, (
                f'crop_size {self.crop_size} exceeds voxel size ({h}, {w}). '
                f'Reduce crop_size or check sensor/ROI dimensions.'
            )
            start_y = (h - crop_h) // 2
            start_x = (w - crop_w) // 2
            voxel_0 = voxel_0[:, start_y:start_y + crop_h, start_x:start_x + crop_w]
            voxel_1 = voxel_1[:, start_y:start_y + crop_h, start_x:start_x + crop_w]

        # [num_bins, H, W] -> torch Tensor
        voxel_0 = torch.from_numpy(voxel_0).float()
        voxel_1 = torch.from_numpy(voxel_1).float()

        return {
            'voxel_0': voxel_0,   # [C, H, W]
            'voxel_1': voxel_1,   # [C, H, W]
            't_start': t_start,
            't_end':   t_end,
        }

    def _load_voxel(self, reader: CustomHDF5Reader, t_start: float, t_end: float,
                    side: str = 'left') -> np.ndarray:
        """Load events in [t_start, t_end], optionally rectify, and convert to voxel [num_bins, H, W]."""
        idx_start = reader.find_ts_index(t_start)
        idx_end   = reader.find_ts_index(t_end)
        events    = reader.get_events(idx_start, idx_end)

        if self.calib is not None:
            # After rectify_events, coords are ROI-local: origin = (0,0), size = (roi_h, roi_w)
            _, _, roi_w, roi_h = self.calib.valid_roi
            voxel_h, voxel_w = roi_h, roi_w

            events = self.calib.rectify_events(events, side)
            # crop only valid events in both left and right cameras, to avoid introducing artifacts from zero-padding
            events = events[(events[:, 0] >= 0) & (events[:, 0] < roi_w) &
                            (events[:, 1] >= 0) & (events[:, 1] < roi_h)]
            if events.shape[0] < 2:
                return np.zeros((self.voxel_bins, voxel_h, voxel_w), dtype=np.float32)
        else:
            voxel_h, voxel_w = self.sensor_h, self.sensor_w

            if events.shape[0] < 2:
                return np.zeros((self.voxel_bins, voxel_h, voxel_w), dtype=np.float32)

        return events_to_voxel(events, self.voxel_bins, voxel_h, voxel_w)
