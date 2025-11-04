#!/bin/bash

# Activate conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate lerobot_pi05

# Set environment variables
export NUM_GPUS=4
export OUTPUT_DIR="/data/output/pi05_task0000"
export BATCH_SIZE=1  # Per GPU batch size (effective batch size = 1 * 4 = 4) - Pi0.5 is memory intensive
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export NUM_STEPS=100000
export SAVE_FREQ=5000
export LOG_FREQ=200
export REPO_ID="ywu67/pi05-task0000"
export DATASET_ID="ywu67/2025-challenge-demos-task0000"
export JOB_NAME="pi05_task0000_100k"

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
  --num_workers=16 \
  --policy.push_to_hub=true \
  --policy.type=pi05 \
  --policy.repo_id=$REPO_ID \
  --policy.paligemma_variant=gemma_2b \
  --policy.action_expert_variant=gemma_300m \
  --policy.max_state_dim=256 \
  --policy.max_action_dim=32 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.dtype=float32 \
  --policy.num_inference_steps=10 \
  --dataset.repo_id=$DATASET_ID \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --job_name=$JOB_NAME
