"""
Rename all .jpg files in the images directory to be sequential (1.jpg, 2.jpg, ...), filling any gaps.
"""
from pathlib import Path
import os

IMAGES_DIR = Path(__file__).parent / "images"

# Get all .jpg files, sorted by their numeric value if possible
files = [f for f in IMAGES_DIR.iterdir() if f.suffix == ".jpg" and f.stem.isdigit()]
files = sorted(files, key=lambda f: int(f.stem))

# Rename to sequential numbers starting from 1
for idx, file in enumerate(files, 1):
    new_name = f"{idx}.jpg"
    new_path = IMAGES_DIR / new_name
    if file.name != new_name:
        # Avoid overwriting by using a temporary name if needed
        tmp_path = IMAGES_DIR / f"tmp_{idx}.jpg"
        file.rename(tmp_path)

# Now rename all tmp_*.jpg to their final names
for idx, _ in enumerate(files, 1):
    tmp_path = IMAGES_DIR / f"tmp_{idx}.jpg"
    final_path = IMAGES_DIR / f"{idx}.jpg"
    if tmp_path.exists():
        tmp_path.rename(final_path)

print("Renamed all images to sequential order.")
