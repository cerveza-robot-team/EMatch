"""Dump raw EMatch disparity + flow + reference frames + timestamps for VO.

Two checkpoints (disparity + flow) are loaded sequentially against the same
EventMatch backbone. For each time window in the EVB_CIRS CustomSequence,
we save:

  disparity/disp_NNNNNN.npy   float32  (H, W)        — pixels (rectified)
  flow/flow_NNNNNN.npy        float32  (H, W, 2)     — (du, dv) pixels
  frames/frame_NNNNNN.png     uint8    (H, W)        — voxel-sum reference
  frames/timestamps.txt                 (one t per row, seconds)
  disparity/pairs.csv                   (idx, ts_l, dt_ms, mean_disp, ...)

Outputs sit at the EventMatch crop resolution. Center-crop offsets:
  start_y = (sensor_h - crop_h) // 2
  start_x = (sensor_w - crop_w) // 2
must be applied to (cx, cy) when running VO with rectified intrinsics.

Usage:
    python dump_ematch_for_vo.py \
        --configs_model    models/configs_model/ematch/disparity.yaml \
        --configs_dataset  datasets/configs_dataset/ematch/evb_cirs/evb_cirs.yaml \
        --disp_ckpt        checkpoints/ematch/disparity/dsec/default/stage2/final.pth \
        --flow_ckpt        checkpoints/ematch/flow/dsec/default/stage2/final.pth \
        --out_dir          /home/vsjay/ifros_hop_ws/output/1_0m_out_ematch
"""

import argparse
import csv
import os

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.EVB_CIRS.CustomSequence_ematch import CustomSequence_ematch
from models.ematch.ematch import EventMatch
from utils.disparity_vis import tensor_to_disparity_jet_image
from utils.flow_viz import flow_tensor_to_image
from utils.EventToImage import voxel_to_rgb


def build_model(model_cfg_path: str, ckpt_path: str, device):
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)
    model = EventMatch(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    weights = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model


def voxel_to_uint8(voxel: torch.Tensor) -> np.ndarray:
    """Sum over temporal bins, normalise to uint8 grey."""
    img = voxel.sum(dim=0).abs().cpu().numpy()
    if img.max() > 0:
        img = (255.0 * (img / img.max())).clip(0, 255)
    return img.astype(np.uint8)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    # --- dataset ---
    with open(args.configs_dataset) as f:
        cfg_ds = yaml.safe_load(f)
    if args.root_path:
        cfg_ds['root_path'] = args.root_path
    if args.sequence_name:
        cfg_ds['SequenceNames'] = [args.sequence_name]
    if args.crop_size:
        cfg_ds['crop_size'] = [int(x) for x in args.crop_size.split('x')]
    if args.stride_ms is not None:
        cfg_ds['stride'] = args.stride_ms
    if args.expected_disp is not None:
        cfg_ds['expected_disparity_at_1m'] = args.expected_disp
    if args.dt_ms is not None:
        cfg_ds['dt'] = args.dt_ms
    seq_name = cfg_ds.get('SequenceNames', ['custom'])[0]
    dataset = CustomSequence_ematch(cfg_ds, seq_name)
    n = len(dataset)
    print(f'[dataset] {seq_name}: {n} windows, crop_size={cfg_ds["crop_size"]}')

    # Disparity-at-1m shift applied during rectify_events: events are shifted
    # so the rig "looks like" expected_disparity at 1m (DSEC scale).
    # Loader does left x -= int(s/2), right x += int(s/2), so the effective
    # additive shift on disparity is 2*int(s/2) (not the raw float s). To
    # recover TRUE rectified-pixel disparity, add exactly that integer back.
    shift_at_1m = float(getattr(dataset.calib, 'shift_at_1m', None) or 0.0) \
        if dataset.calib is not None else 0.0
    shift_applied = 2 * int(shift_at_1m / 2)
    print(f'[shift] disparity_at_1m correction: +{shift_applied} px '
          f'(raw shift_at_1m={shift_at_1m:.2f} px, integer-applied by loader)')

    # --- output dirs ---
    disp_dir = os.path.join(args.out_dir, 'disparity')
    flow_dir = os.path.join(args.out_dir, 'flow')
    frame_dir = os.path.join(args.out_dir, 'frames')
    voxel_dir = os.path.join(args.out_dir, 'voxel')
    disp_viz_dir = os.path.join(disp_dir, 'viz')
    flow_viz_dir = os.path.join(flow_dir, 'viz')
    for d in (disp_dir, flow_dir, frame_dir, voxel_dir, disp_viz_dir, flow_viz_dir):
        os.makedirs(d, exist_ok=True)

    # ---------------- disparity pass (also writes ref frames + timestamps) ---
    print(f'[stage] disparity inference  ckpt={args.disp_ckpt}')
    model = build_model(args.configs_model_disp, args.disp_ckpt, device)
    pairs_rows = []
    mean_disps = []
    ts_all = []
    ts_file = open(os.path.join(frame_dir, 'timestamps.txt'), 'w')
    with torch.no_grad():
        for i in tqdm(range(n)):
            s = dataset[i]
            v0 = s['voxel_0'][None].to(device)
            v1 = s['voxel_1'][None].to(device)

            # ref frame from voxel sum (left), written once during disp pass
            cv2.imwrite(
                os.path.join(frame_dir, f'frame_{i:06d}.png'),
                voxel_to_uint8(s['voxel_0']),
            )
            # voxel RGB viz for both eyes (for Rerun + manual inspection)
            if args.save_voxel_viz:
                for side, vox in (('left', s['voxel_0']), ('right', s['voxel_1'])):
                    rgb = voxel_to_rgb(vox.cpu().numpy()).transpose(1, 2, 0)  # (3,H,W) -> (H,W,3)
                    cv2.imwrite(
                        os.path.join(voxel_dir, f'voxel_{i:06d}_{side}.png'),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    )
            ts = float(s['t_end'])
            ts_all.append(ts)
            ts_file.write(f'{ts:.9f}\n')

            out = model(v0, v1, task='disparity')
            disp_t = out['disparity_preds'][-1][0].cpu()
            disp = disp_t.numpy().astype(np.float32)
            # undo the 1m shift applied to events during rectification
            if shift_applied != 0:
                disp = disp + np.float32(shift_applied)
            np.save(os.path.join(disp_dir, f'disp_{i:06d}.npy'), disp)
            # also a colormap jpg for quick inspection (clipped at --disp_vmax,
            # which should now scale to TRUE disp range, not the shifted one)
            import torch as _torch
            img = tensor_to_disparity_jet_image(
                _torch.from_numpy(disp), vmax=args.disp_vmax
            )
            img.save(os.path.join(disp_viz_dir, f'viz_{i:06d}.jpg'))
            mean_disp = float(np.nanmean(disp[disp > 0])) if np.any(disp > 0) else 0.0
            pairs_rows.append(dict(
                idx=i,
                l_path=f'frames/frame_{i:06d}.png',
                r_path=f'frames/frame_{i:06d}.png',
                ts_l=ts, ts_r=ts, dt_ms=0.0,
                mean_disp=mean_disp, pct_valid=100.0,
            ))
            mean_disps.append(mean_disp)
    ts_file.close()
    del model
    torch.cuda.empty_cache()
    print(f'[disparity] mean(mean_disp) = {float(np.mean(mean_disps)):.2f} px')
    with open(os.path.join(disp_dir, 'pairs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs_rows[0].keys()))
        w.writeheader()
        w.writerows(pairs_rows)
    print(f'[disparity] wrote {n} disp + pairs.csv -> {disp_dir}')

    # ---------------- flow pass (re-reads dataset windows) ----------------
    print(f'[stage] flow inference       ckpt={args.flow_ckpt}')
    model = build_model(args.configs_model_flow, args.flow_ckpt, device)
    mean_mags = []
    with torch.no_grad():
        for i in tqdm(range(n)):
            s = dataset[i]
            v0 = s['voxel_0'][None].to(device)
            v1 = s['voxel_1'][None].to(device)
            out = model(v0, v1, task='flow')
            flow_t = out['flow_preds'][-1][0].cpu()      # [2, H, W]
            flow = np.transpose(flow_t.numpy().astype(np.float32), (1, 2, 0))
            np.save(os.path.join(flow_dir, f'flow_{i:06d}.npy'), flow)
            # colormap PNG
            rgb = flow_tensor_to_image(flow_t).transpose(1, 2, 0)
            cv2.imwrite(
                os.path.join(flow_viz_dir, f'viz_{i:06d}.png'),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            mean_mags.append(float(np.mean(np.hypot(flow[..., 0], flow[..., 1]))))
    del model
    torch.cuda.empty_cache()
    print(f'[flow] mean(|flow|) = {float(np.mean(mean_mags)):.3f} px')
    print(f'[flow] wrote {n} flow npys -> {flow_dir}')

    # also dump a small summary
    summary = dict(
        n_windows=n,
        crop_size=list(cfg_ds['crop_size']),
        sensor=[cfg_ds['sensor_height'], cfg_ds['sensor_width']],
        dt_ms=cfg_ds.get('dt'),
        stride_ms=cfg_ds.get('stride'),
        alpha_rectify=cfg_ds.get('alpha', 0.5),
        expected_disparity_at_1m=cfg_ds.get('expected_disparity_at_1m'),
        shift_at_1m=shift_at_1m,
        shift_applied=shift_applied,
        disp_units='true rectified disparity (model output + shift_applied)',
        disp_ckpt=args.disp_ckpt,
        flow_ckpt=args.flow_ckpt,
        mean_disp_avg=float(np.mean(mean_disps)),
        mean_flow_mag_avg=float(np.mean(mean_mags)),
    )
    with open(os.path.join(args.out_dir, 'ematch_summary.yaml'), 'w') as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f'[done] summary -> {args.out_dir}/ematch_summary.yaml')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs_dataset', required=True,
                   help='EVB_CIRS dataset YAML')
    p.add_argument('--configs_model_disp', default='models/configs_model/ematch/disparity.yaml',
                   help='Model config for disparity task')
    p.add_argument('--configs_model_flow', default='models/configs_model/ematch/flow.yaml',
                   help='Model config for flow task')
    p.add_argument('--disp_ckpt', required=True,
                   help='Checkpoint .pth for disparity head')
    p.add_argument('--flow_ckpt', required=True,
                   help='Checkpoint .pth for flow head')
    p.add_argument('--out_dir', required=True,
                   help='Output root (will contain disparity/, flow/, frames/)')
    p.add_argument('--root_path', default=None,
                   help='Override root_path (HDF5 path) from the dataset YAML')
    p.add_argument('--sequence_name', default=None,
                   help='Override SequenceNames[0] from the dataset YAML')
    p.add_argument('--crop_size', default=None,
                   help='Override crop_size as "HxW" e.g. 512x960')
    p.add_argument('--stride_ms', type=int, default=None,
                   help='Override window stride in ms (default = dataset YAML)')
    p.add_argument('--dt_ms', type=int, default=None,
                   help='Override window dt (event accumulation, ms)')
    p.add_argument('--expected_disp', type=float, default=None,
                   help='Override expected_disparity_at_1m (px). Sets the '
                        'event x-shift target; raise toward the training '
                        'distributions disparity at 1m (DSEC ~128, MVSEC ~5).')
    p.add_argument('--disp_vmax', type=float, default=600.0,
                   help='Max disparity for jet colormap (px). Set near the '
                        'p95 of your disp output so the map uses the range.')
    p.add_argument('--save_voxel_viz', action='store_true', default=True,
                   help='Dump voxel_to_rgb PNGs for left+right under voxel/ '
                        '(used by log_to_rerun.py).')
    p.add_argument('--no_voxel_viz', dest='save_voxel_viz', action='store_false',
                   help='Disable voxel RGB dump (saves ~50 MB / 500 frames).')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # CustomSequence reads root_path relative to PWD; user should cd into EMatch/
    main(args)
