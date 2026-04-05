"""
Remove duplicate images from the dataset and their corresponding labels.

Uses frame-diff approach (like comparing video frames):
  - Iterates images sequentially
  - Compares each frame to the previous *kept* frame via cv2.absdiff
  - If less than threshold% of pixels changed → duplicate, skip it

Usage:
    python remove_duplicates.py --image_dir images --results_dir results --dry_run
    python remove_duplicates.py --image_dir images --results_dir results --threshold 0.02 --pixel_diff 30
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate images (frame-diff)")
    parser.add_argument("--image_dir", required=True, help="Directory with images")
    parser.add_argument("--results_dir", required=True, help="Directory with labels.txt")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Fraction of frame that must change to count as different (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--pixel_diff",
        type=int,
        default=30,
        help="Per-pixel intensity difference to count as changed (default: 30)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report duplicates without deleting",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    labels_path = results_dir / "labels.txt"

    # Load images in numeric order (1.jpg, 2.jpg, ...)
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: int(p.stem),
    )
    print(f"Found {len(image_files)} images")

    # Load labels
    labels = labels_path.read_text().strip().splitlines()
    assert len(labels) == len(image_files), f"Label count {len(labels)} != image count {len(image_files)}"

    # --- Frame-diff duplicate detection (like video stream) ---
    prev_frame = None
    keep_indices = []
    duplicate_indices = []

    for i, img_path in enumerate(tqdm(image_files, desc="Frame-diff scanning")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            duplicate_indices.append(i)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            changed = (diff > args.pixel_diff).sum() / diff.size

            if changed > args.threshold:
                # Different enough — keep this frame
                keep_indices.append(i)
                prev_frame = gray
            else:
                # Too similar to previous kept frame — duplicate
                duplicate_indices.append(i)
        else:
            # First frame — always keep
            keep_indices.append(i)
            prev_frame = gray

    print(f"\nImages to keep: {len(keep_indices)}")
    print(f"Images to remove: {len(duplicate_indices)}")

    removed_labels = defaultdict(int)
    for idx in duplicate_indices:
        removed_labels[labels[idx]] += 1
    print(f"Removed image labels: {dict(removed_labels)}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")
        # Show some examples of removed duplicates
        shown = 0
        for idx in duplicate_indices[:10]:
            print(f"  Remove {image_files[idx].name} (label: {labels[idx]})")
            shown += 1
        if len(duplicate_indices) > 10:
            print(f"  ... and {len(duplicate_indices) - 10} more")
        return

    # Delete duplicate images
    for idx in duplicate_indices:
        image_files[idx].unlink()

    # Build new labels
    new_labels = [labels[i] for i in keep_indices]

    # Rename remaining images to 1.jpg, 2.jpg, ...
    remaining_files = [image_files[i] for i in keep_indices]
    for i, f in enumerate(remaining_files):
        f.rename(image_dir / f"_temp_{i}.jpg")
    for i in range(len(remaining_files)):
        (image_dir / f"_temp_{i}.jpg").rename(image_dir / f"{i + 1}.jpg")

    # Write updated labels
    labels_path.write_text("\n".join(new_labels) + "\n")

    print(f"\nDone! Kept {len(new_labels)} images, renumbered 1.jpg to {len(new_labels)}.jpg")

    final_dist = defaultdict(int)
    for l in new_labels:
        final_dist[l] += 1
    print(f"Final label distribution: {dict(final_dist)}")


if __name__ == "__main__":
    main()
