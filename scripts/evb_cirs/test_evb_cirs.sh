#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/disparity.yaml"
# MODEL="models/configs_model/ematch/flow.yaml"

# Disparity
# CHECKPOINT="checkpoints/ematch/disparity/mvsec_split3/dsecUnifiedBase/final.pth"
CHECKPOINT="checkpoints/ematch/disparity/dsec/default/stage2/final.pth"

# Flow
# CHECKPOINT="checkpoints/ematch/flow/dsec/default/stage2/final.pth"


DATASET="datasets/configs_dataset/ematch/evb_cirs/evb_cirs.yaml"
SAVE_DIR="outcomes/ematch/custom/CIRS_EVB/default"

mkdir -p ${SAVE_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python3 test_evb_cirs_ematch.py \
--configs_model ${MODEL} \
--checkpoint ${CHECKPOINT} \
--configs_dataset ${DATASET} \
--save_dir ${SAVE_DIR} \
--event_voxel \
--task disparity \
2>&1 | tee -a ${SAVE_DIR}/test.log