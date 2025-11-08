#!/usr/bin/env python3
"""
Example: Using Image Tiling with SmolVLA

This script demonstrates how to enable and use the image tiling feature
in SmolVLA for improved visual detail processing.
"""

import torch
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy


def example_without_tiling():
    """Standard SmolVLA configuration without tiling (default)."""
    print("\n=== Example 1: Standard Configuration (No Tiling) ===")

    config = SmolVLAConfig(
        do_image_splitting=False,  # Disabled (default)
        resize_imgs_with_padding=(512, 512),
        max_tile_size=512,
    )

    print(f"Image Tiling: {config.do_image_splitting}")
    print(f"Input Size: {config.resize_imgs_with_padding}")
    print(f"Expected Visual Tokens per Image: ~256")
    print(f"Inference Speed: Fast (baseline)")
    print(f"Use Case: Real-time robotics, standard cameras")


def example_with_tiling():
    """SmolVLA configuration with image tiling enabled."""
    print("\n=== Example 2: With Image Tiling ===")

    config = SmolVLAConfig(
        do_image_splitting=True,  # Enable tiling
        resize_imgs_with_padding=(1024, 1024),  # Higher resolution
        max_tile_size=512,
    )

    print(f"Image Tiling: {config.do_image_splitting}")
    print(f"Input Size: {config.resize_imgs_with_padding}")
    print(f"Tile Size: {config.max_tile_size}x{config.max_tile_size}")
    print(f"Number of Tiles: 4 tiles + 1 global = 5 images")
    print(f"Expected Visual Tokens: ~1280 (5 × 256)")
    print(f"Inference Speed: ~4-5x slower than baseline")
    print(f"Use Case: High-res cameras, fine manipulation tasks")


def example_high_resolution_tiling():
    """SmolVLA with aggressive tiling for very high resolution."""
    print("\n=== Example 3: High Resolution Tiling ===")

    config = SmolVLAConfig(
        do_image_splitting=True,
        resize_imgs_with_padding=(2048, 2048),  # Very high resolution
        max_tile_size=512,
    )

    # Calculate expected tiles
    num_tiles = (2048 // 512) ** 2  # 4x4 grid
    total_images = num_tiles + 1  # +1 for global view
    total_tokens = total_images * 256

    print(f"Image Tiling: {config.do_image_splitting}")
    print(f"Input Size: {config.resize_imgs_with_padding}")
    print(f"Tile Grid: 4x4 (16 tiles)")
    print(f"Total Images: {total_images} (16 tiles + 1 global)")
    print(f"Expected Visual Tokens: ~{total_tokens}")
    print(f"Inference Speed: ~15-20x slower than baseline")
    print(f"Use Case: Ultra-fine manipulation, small object detection")


def example_custom_tile_size():
    """Custom tile size for different tiling strategies."""
    print("\n=== Example 4: Custom Tile Size ===")

    config = SmolVLAConfig(
        do_image_splitting=True,
        resize_imgs_with_padding=(1024, 1024),
        max_tile_size=256,  # Smaller tiles
    )

    # Calculate expected tiles
    num_tiles = (1024 // 256) ** 2  # 4x4 grid
    total_images = num_tiles + 1
    total_tokens = total_images * 256

    print(f"Image Tiling: {config.do_image_splitting}")
    print(f"Input Size: {config.resize_imgs_with_padding}")
    print(f"Tile Size: {config.max_tile_size}x{config.max_tile_size}")
    print(f"Tile Grid: 4x4 (16 tiles)")
    print(f"Total Images: {total_images}")
    print(f"Expected Visual Tokens: ~{total_tokens}")
    print(f"Trade-off: More tiles = better detail but slower inference")


def example_programmatic_toggle():
    """Toggle tiling on/off programmatically."""
    print("\n=== Example 5: Programmatic Toggle ===")

    # Create policy with tiling
    config = SmolVLAConfig(
        do_image_splitting=True,
        resize_imgs_with_padding=(1024, 1024),
    )
    policy = SmolVLAPolicy(config)

    print(f"Initial: Tiling = {policy.config.do_image_splitting}")

    # Disable for fast inference
    policy.config.do_image_splitting = False
    print(f"Disabled for inference: Tiling = {policy.config.do_image_splitting}")

    # Re-enable if needed
    policy.config.do_image_splitting = True
    print(f"Re-enabled: Tiling = {policy.config.do_image_splitting}")

    print("\nUse Case: Train with tiling, deploy without for speed")


def example_training_command():
    """Example training commands with tiling."""
    print("\n=== Example 6: Training Commands ===")

    print("\n# Train without tiling (fast, baseline):")
    print("""
lerobot-train \\
  --policy.type=smolvla \\
  --policy.do_image_splitting=false \\
  --dataset.repo_id=your/dataset \\
  --batch_size=64 \\
  --steps=100000
""")

    print("\n# Train with tiling (better detail, slower):")
    print("""
lerobot-train \\
  --policy.type=smolvla \\
  --policy.do_image_splitting=true \\
  --policy.max_tile_size=512 \\
  --policy.resize_imgs_with_padding="[1024, 1024]" \\
  --dataset.repo_id=your/dataset \\
  --batch_size=16 \\
  --steps=100000
""")

    print("\nNote: Reduce batch_size when using tiling due to increased memory usage")


def example_performance_comparison():
    """Compare performance characteristics."""
    print("\n=== Example 7: Performance Comparison ===")

    configs = [
        ("Standard (512×512, no tiling)", False, 512, (512, 512)),
        ("Medium (1024×1024, tiling)", True, 512, (1024, 1024)),
        ("High (2048×2048, tiling)", True, 512, (2048, 2048)),
    ]

    print("\n{:<35} {:>10} {:>15} {:>15}".format(
        "Configuration", "Images", "Visual Tokens", "Rel. Speed"
    ))
    print("-" * 80)

    for name, tiling, tile_size, img_size in configs:
        if tiling:
            num_tiles = (img_size[0] // tile_size) ** 2 + 1
            tokens = num_tiles * 256
            speed = f"~{num_tiles}x slower"
        else:
            num_tiles = 1
            tokens = 256
            speed = "Baseline"

        print(f"{name:<35} {num_tiles:>10} {tokens:>15} {speed:>15}")


def main():
    """Run all examples."""
    print("=" * 80)
    print("SmolVLA Image Tiling Examples")
    print("=" * 80)

    example_without_tiling()
    example_with_tiling()
    example_high_resolution_tiling()
    example_custom_tile_size()
    example_programmatic_toggle()
    example_training_command()
    example_performance_comparison()

    print("\n" + "=" * 80)
    print("For detailed documentation, see: IMAGE_TILING_GUIDE.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
