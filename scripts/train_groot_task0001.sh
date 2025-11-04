#!/bin/bash

# Activate conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

# Set environment variables
export NUM_GPUS=4
export OUTPUT_DIR="/data/output/groot_task0000"
export BATCH_SIZE=8  # Per GPU batch size (effective batch size = 8 * 4 = 32)
export NUM_STEPS=100000
export SAVE_FREQ=5000
export LOG_FREQ=200
export REPO_ID="ywu67/groot-task0000"
export DATASET_ID="ywu67/2025-challenge-demos-task0000"
export JOB_NAME="groot_task0000_100k"

# Using a multi-GPU setup
accelerate launch \
  --multi_gpu \
  --num_processes=$NUM_GPUS \
  $(which lerobot-train) \
  --output_dir=$OUTPUT_DIR \
  --save_checkpoint=true \
  --batch_size=$BATCH_SIZE \
  --steps=$NUM_STEPS \
  --save_freq=$SAVE_FREQ \
  --log_freq=$LOG_FREQ \
  --policy.push_to_hub=true \
  --policy.type=groot \
  --policy.repo_id=$REPO_ID \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id=$DATASET_ID \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --job_name=$JOB_NAME
