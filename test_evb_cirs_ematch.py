"""
Inference script for EMatch on a custom stereo event camera dataset.

Supports both 'disparity' and 'flow' tasks.
No ground truth is required — outputs are saved as images.

Usage:
    python test_custom_ematch.py \
        --configs_model  models/configs_model/ematch/disparity.yaml \
        --configs_dataset datasets/configs_dataset/ematch/custom/eval.yaml \
        --checkpoint     checkpoints/ematch_disparity.pth \
        --task           disparity \
        --save_dir       results/custom/

    python test_custom_ematch.py \
        --configs_model  models/configs_model/ematch/flow.yaml \
        --configs_dataset datasets/configs_dataset/ematch/custom/eval.yaml \
        --checkpoint     checkpoints/ematch_flow.pth \
        --task           flow \
        --save_dir       results/custom/
"""

import argparse
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


def get_args_parser():
    parser = argparse.ArgumentParser(description='EMatch inference on custom dataset')
    parser.add_argument('--seed',            default=0,            type=int)
    parser.add_argument('--save_dir',        default='results/custom', type=str,
                        help='Directory to save output images')
    parser.add_argument('--task',            default='disparity',  type=str,
                        choices=['disparity', 'flow'],
                        help='Prediction task: disparity or flow')
    parser.add_argument('--configs_model',   required=True,        type=str,
                        help='Path to model YAML config')
    parser.add_argument('--configs_dataset', required=True,        type=str,
                        help='Path to dataset YAML config')
    parser.add_argument('--checkpoint',      required=True,        type=str,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--disp_vmax',       default=64,           type=float,
                        help='Max disparity value for colormap scaling')
    parser.add_argument('--event_voxel',     default=False,        action='store_true',
                        help='Also save voxel grid visualizations')
    return parser


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on: {device}')

    # ---- Model ----
    with open(args.configs_model, 'r') as f:
        model_config = yaml.safe_load(f)
    model = EventMatch(model_config).to(device)
    print(f'Loaded model config from {args.configs_model}')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False)
    model.eval()
    print(f'Loaded checkpoint from {args.checkpoint}')

    # ---- Dataset ----
    with open(args.configs_dataset, 'r') as f:
        config_dataset = yaml.safe_load(f)

    sequence_names = config_dataset.get('SequenceNames', ['custom'])
    datasets = []
    for name in sequence_names:
        datasets.append((name, CustomSequence_ematch(config_dataset, name)))

    # ---- Inference ----
    for seq_name, dataset in datasets:
        print(f'\nProcessing sequence: {seq_name}  ({len(dataset)} windows)')

        visual_path = os.path.join(args.save_dir, 'visual', seq_name)
        npy_path = os.path.join(args.save_dir, 'npy', seq_name)
        os.makedirs(visual_path, exist_ok=True)
        os.makedirs(npy_path, exist_ok=True)

        if args.event_voxel:
            voxel_path = os.path.join(args.save_dir, 'voxel', seq_name)
            os.makedirs(voxel_path, exist_ok=True)

        # Pre-rectification disparity shift applied by loader (left x -= s/2, right x += s/2,
        # both integer-rounded). Net effect on disparity: d_input = d_true - s_applied,
        # where s_applied = 2 * int(shift_at_1m / 2). Undo here so saved/visualized
        # disparity is in true rectified-image coordinates.
        s_applied = 0
        if args.task == 'disparity':
            calib = getattr(dataset, 'calib', None)
            shift = getattr(calib, 'shift_at_1m', None) if calib is not None else None
            if shift is not None:
                s_applied = 2 * int(shift / 2)
                print(f'  Undoing pre-rectification disparity shift: +{s_applied} px')

        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                sample = dataset[i]
                voxel_0 = sample['voxel_0'][None].to(device)  # [1, C, H, W]
                voxel_1 = sample['voxel_1'][None].to(device)

                # Optional: save voxel visualizations
                if args.event_voxel:
                    from utils.EventToImage import voxel_to_rgb
                    for side, vox in [('left', voxel_0), ('right', voxel_1)]:
                        rgb = voxel_to_rgb(vox[0].cpu().numpy()).transpose(1, 2, 0)
                        cv2.imwrite(
                            os.path.join(voxel_path, f'{str(i).zfill(6)}_{side}.png'),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        )

                # Run model
                results_dict = model(voxel_0, voxel_1, task=args.task)

                if args.task == 'disparity':
                    pred = results_dict['disparity_preds'][-1][0].cpu()  # [H, W]
                    if s_applied != 0:
                        pred = pred + s_applied
                    pred_np = pred.numpy().astype(np.float32)
                    np.save(os.path.join(npy_path, f'{str(i).zfill(6)}.npy'), pred_np)
                    img = tensor_to_disparity_jet_image(pred, vmax=args.disp_vmax)
                    img.save(os.path.join(visual_path, f'{str(i).zfill(6)}.jpg'))

                elif args.task == 'flow':
                    pred = results_dict['flow_preds'][-1][0].cpu()  # [2, H, W]
                    pred_np = pred.numpy().astype(np.float32)
                    np.save(os.path.join(npy_path, f'{str(i).zfill(6)}.npy'), pred_np)
                    rgb = flow_tensor_to_image(pred).transpose(1, 2, 0)
                    cv2.imwrite(
                        os.path.join(visual_path, f'{str(i).zfill(6)}.png'),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    )

        print(f'Saved predictions to {visual_path}')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
