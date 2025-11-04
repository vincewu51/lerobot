#!/usr/bin/env python3
"""
Upload all checkpoints to Hugging Face Hub.
Each checkpoint will be uploaded as a separate revision.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "ywu67/smolvla-task0000"
CHECKPOINTS_DIR = Path("/data/output/smolvla_task0000/checkpoints")

def upload_checkpoints():
    api = HfApi()

    # Create repo if it doesn't exist (will not fail if it already exists)
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repository {REPO_ID} is ready")
    except Exception as e:
        print(f"Note: {e}")

    # Get all checkpoint directories (excluding 'last' symlink)
    checkpoints = sorted([d for d in CHECKPOINTS_DIR.iterdir()
                         if d.is_dir() and d.name.isdigit()])

    print(f"\nFound {len(checkpoints)} checkpoints to upload:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")

    # Upload each checkpoint
    for ckpt_dir in checkpoints:
        step = ckpt_dir.name
        pretrained_model_dir = ckpt_dir / "pretrained_model"

        if not pretrained_model_dir.exists():
            print(f"\nSkipping {step}: no pretrained_model directory")
            continue

        print(f"\n{'='*60}")
        print(f"Uploading checkpoint: step {step}")
        print(f"{'='*60}")

        try:
            # Upload the pretrained model to a branch named after the checkpoint
            commit_message = f"Add checkpoint at step {step}"

            # Upload folder to the main branch with a prefix
            api.upload_folder(
                folder_path=str(pretrained_model_dir),
                repo_id=REPO_ID,
                repo_type="model",
                path_in_repo=f"checkpoints/step_{step}",
                commit_message=commit_message,
            )

            print(f"✓ Successfully uploaded checkpoint {step}")
            print(f"  Location: {REPO_ID}/checkpoints/step_{step}")

        except Exception as e:
            print(f"✗ Error uploading checkpoint {step}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Upload complete!")
    print(f"View your model at: https://huggingface.co/{REPO_ID}")
    print(f"{'='*60}")

if __name__ == "__main__":
    upload_checkpoints()
