"""Re-render disparity viz JPGs from existing disp_*.npy with a higher vmax.

Default disp_vmax was 128 px, but true EVB_CIRS disparities reach 300-700 px
after the 1m-shift undo, so the JET colormap saturates to solid red. This
reads the raw .npy values and rewrites viz_*.jpg at a sensible vmax.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from utils.disparity_vis import tensor_to_disparity_jet_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--disp_dir', required=True,
                    help='Directory containing disp_*.npy (e.g. output/<seq>_ematch/disparity)')
    ap.add_argument('--vmax', type=float, default=600.0)
    ap.add_argument('--out_dir', default=None,
                    help='Where to write viz_*.jpg (default: <disp_dir>/viz)')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.disp_dir, 'viz')
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.disp_dir, 'disp_*.npy')))
    print(f'[viz] {len(paths)} disparity files  vmax={args.vmax}  out={out_dir}')

    for p in paths:
        disp = np.load(p).astype(np.float32)
        disp = np.where(np.isfinite(disp) & (disp > 0), disp, 0.0).astype(np.float32)
        idx = os.path.basename(p).replace('disp_', '').replace('.npy', '')
        img = tensor_to_disparity_jet_image(torch.from_numpy(disp), vmax=args.vmax)
        img.save(os.path.join(out_dir, f'viz_{idx}.jpg'))
    print(f'[viz] wrote {len(paths)} files')


if __name__ == '__main__':
    main()
