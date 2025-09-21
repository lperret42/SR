#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
from pathlib import Path
import json
import random


def load_metadata(root_dir):
    meta_path = Path(root_dir) / 'metadata.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None


def infer_lr_dir(root_dir, metadata):
    root = Path(root_dir)
    if metadata and 'downsample_factor' in metadata:
        d = metadata['downsample_factor']
        candidate = root / f"lr_x{d}"
        if candidate.exists():
            return candidate
    # Fallback: pick first lr_x* directory
    for p in root.glob('lr_x*'):
        if p.is_dir():
            return p
    raise FileNotFoundError("Could not locate an lr_x* directory")


def load_image_pairs(dataset_dir, shuffle=False, seed=42):
    dataset_path = Path(dataset_dir)
    sr_dir = dataset_path / "sr"
    if not sr_dir.exists():
        raise FileNotFoundError(f"SR directory not found: {sr_dir}")

    metadata = load_metadata(dataset_dir)
    lr_dir = infer_lr_dir(dataset_dir, metadata)

    # Collect all sr tiles recursively
    sr_files = sorted([p for p in sr_dir.rglob('*.png')])
    pairs = []
    for sr_file in sr_files:
        rel = sr_file.relative_to(sr_dir)  # e.g. sceneA/sceneA_tile_0000_x...png
        lr_file = lr_dir / rel
        if lr_file.exists():
            # Pair order expected later: (sr_path, lr_path)
            pairs.append((str(sr_file), str(lr_file)))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(pairs)
    return pairs, metadata, lr_dir


def create_comparison_view(sr_path, lr_path, canvas_color=(0, 0, 0), separator_width=10):
    """Create side-by-side comparison: (1) LR native (centered on black canvas sized like SR)
    (2) bicubic upsample of LR to SR size, (3) SR original"""
    # Load images (BGR)
    lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
    if lr_img is None or sr_img is None:
        raise RuntimeError("Failed to load one of the images")

    sr_h, sr_w = sr_img.shape[:2]
    lr_h, lr_w = lr_img.shape[:2]

    # Canvas for native LR (keep original size centered)
    canvas = np.zeros((sr_h, sr_w, 3), dtype=np.uint8)
    if canvas_color != (0, 0, 0):
        canvas[:] = canvas_color
    # Center coordinates
    y_off = (sr_h - lr_h) // 2
    x_off = (sr_w - lr_w) // 2
    canvas[y_off:y_off+lr_h, x_off:x_off+lr_w] = lr_img

    # Bicubic upscale
    lr_bicubic = cv2.resize(lr_img, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)

    # Separator
    separator = np.ones((sr_h, separator_width, 3), dtype=np.uint8) * 255

    comparison = np.concatenate([
        canvas, separator,
        lr_bicubic, separator,
        sr_img
    ], axis=1)
    return comparison


def add_labels(img, labels, separator_width=10, font_scale=0.7, thickness=2):
    h, w = img.shape[:2]
    # We have 3 image sections and 2 separators: total width = 3*section + 2*sep
    # Solve for section width: approximate by measuring separators we inserted.
    # Simpler: compute each section width from height & known order
    # We'll split by separators position heuristically.
    # Instead: pass labels at approximate fractional positions.
    section_w = (w - 2 * separator_width) // 3
    x_positions = [
        section_w // 2,
        section_w + separator_width + section_w // 2,
        2 * (section_w + separator_width) + section_w // 2
    ]
    for label, x in zip(labels, x_positions):
        cv2.putText(img, label, (int(x - 60), 30), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)
    return img


def print_metadata_summary(metadata, pairs_count):
    if not metadata:
        print("No metadata.json found. Showing basic info only.")
        print(f"Pairs found: {pairs_count}")
        return
    keys = [
        'requested_images', 'selected_images', 'n_tiles_per_image', 'tile_size',
        'downsample_factor', 'total_sr_tiles', 'total_lr_tiles'
    ]
    print("Dataset metadata summary:")
    for k in keys:
        if k in metadata:
            print(f"  {k}: {metadata[k]}")
    if 'degradations' in metadata:
        print(f"  degradations: {metadata['degradations']}")
    if 'time' in metadata and 'duration_seconds' in metadata['time']:
        print(f"  generation_time_s: {metadata['time']['duration_seconds']}")
    print(f"  pairs_available: {pairs_count}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SR/LR tile pairs in new dataset format")
    parser.add_argument('dataset_dir', type=str, help='Path to dataset root (containing sr/ and lr_x*/ )')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle order of pairs')
    parser.add_argument('--seed', type=int, default=42, help='Seed used when shuffling')
    parser.add_argument('--separator_width', type=int, default=10, help='Separator width between panels')
    args = parser.parse_args()

    pairs, metadata, lr_dir = load_image_pairs(args.dataset_dir, shuffle=args.shuffle, seed=args.seed)
    print_metadata_summary(metadata, len(pairs))

    if len(pairs) == 0:
        print('No image pairs found!')
        return

    print('\nControls:')
    print('  a: Previous')
    print('  z: Next')
    print('  q/ESC: Quit')
    print('  r: Reset to first')
    if args.shuffle:
        print(f"  Order: shuffled (seed={args.seed})")
    else:
        print("  Order: sequential")

    current_idx = 0
    total = len(pairs)

    while True:
        sr_path, lr_path = pairs[current_idx]
        try:
            comparison = create_comparison_view(sr_path, lr_path, separator_width=args.separator_width)
        except RuntimeError as e:
            print(f"Error loading images for pair index {current_idx}: {e}")
            current_idx = (current_idx + 1) % total
            continue

        labels = ["LR Native (centered)", "LR Bicubic", "SR"]
        comparison = add_labels(comparison, labels, separator_width=args.separator_width)

        counter_text = f"Pair {current_idx + 1}/{total} - {Path(sr_path).name}"
        cv2.putText(comparison, counter_text, (10, comparison.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('SR Dataset Viewer', comparison)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('z'):
            current_idx = (current_idx + 1) % total
        elif key == ord('a'):
            current_idx = (current_idx - 1) % total
        elif key == ord('r'):
            current_idx = 0

    cv2.destroyAllWindows()
    print('Viewer closed.')


if __name__ == '__main__':
    main()
