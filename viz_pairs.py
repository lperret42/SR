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


def discover_lr_factors(root_dir, metadata):
    root = Path(root_dir)
    factors = []
    # Prefer metadata if available
    if metadata:
        if 'downsample_factors' in metadata:
            factors = list(metadata['downsample_factors'])
        elif 'downsample_factor' in metadata:
            factors = [metadata['downsample_factor']]
    if not factors:
        # Fallback: parse directory names lr_x<number>
        for p in root.glob('lr_x*'):
            if p.is_dir():
                try:
                    fct = int(p.name.split('lr_x')[-1])
                    factors.append(fct)
                except ValueError:
                    continue
    # Deduplicate & sort ascending
    uniq = []
    for f in sorted(factors):
        if f not in uniq:
            uniq.append(f)
    if not uniq:
        raise FileNotFoundError("No LR factor directories found (lr_x*)")
    return uniq


def load_image_entries(dataset_dir, shuffle=False, seed=42, require_all=True):
    """Return list of entries: (sr_path, {factor: lr_path})
    require_all=True means skip SR tiles missing any LR factor."""
    dataset_path = Path(dataset_dir)
    sr_dir = dataset_path / 'sr'
    if not sr_dir.exists():
        raise FileNotFoundError(f'SR directory not found: {sr_dir}')
    metadata = load_metadata(dataset_dir)
    factors = discover_lr_factors(dataset_dir, metadata)

    # Build map factor->dir
    lr_dirs = {f: dataset_path / f'lr_x{f}' for f in factors}
    for f, d in lr_dirs.items():
        if not d.exists():
            raise FileNotFoundError(f'LR directory missing for factor {f}: {d}')

    sr_files = sorted(sr_dir.rglob('*.png'))
    entries = []
    for sr_file in sr_files:
        rel = sr_file.relative_to(sr_dir)
        lr_paths = {}
        missing = False
        for f, d in lr_dirs.items():
            candidate = d / rel
            if candidate.exists():
                lr_paths[f] = str(candidate)
            else:
                missing = True
                if require_all:
                    break
        if missing and require_all:
            continue
        if lr_paths:  # ensure at least one
            entries.append((str(sr_file), lr_paths))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(entries)
    return entries, metadata, factors


def create_composite(sr_path, lr_paths_dict, factors, canvas_color=(0,0,0), separator_width=10):
    """Create composite image: for each factor -> native centered + bicubic, then SR.
    Panel order: [LR xF native, LR xF bicubic]* + SR
    Each panel has SR spatial size; native LR is centered on black canvas."""
    sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
    if sr_img is None:
        raise RuntimeError('Failed to load SR tile')
    sr_h, sr_w = sr_img.shape[:2]

    panels = []
    labels = []
    for f in factors:
        lr_path = lr_paths_dict.get(f)
        if lr_path is None:
            continue
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        if lr_img is None:
            continue
        lr_h, lr_w = lr_img.shape[:2]
        # Native centered canvas
        native_canvas = np.zeros((sr_h, sr_w, 3), dtype=np.uint8)
        if canvas_color != (0,0,0):
            native_canvas[:] = canvas_color
        y_off = (sr_h - lr_h) // 2
        x_off = (sr_w - lr_w) // 2
        native_canvas[y_off:y_off+lr_h, x_off:x_off+lr_w] = lr_img
        panels.append(native_canvas)
        labels.append(f'LR x{f} native')
        # Bicubic upscale
        bicubic = cv2.resize(lr_img, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
        panels.append(bicubic)
        labels.append(f'LR x{f} bicubic')
    panels.append(sr_img)
    labels.append('SR')

    # Insert separators
    sep = np.ones((sr_h, separator_width, 3), dtype=np.uint8) * 255
    composite_parts = []
    for idx, p in enumerate(panels):
        composite_parts.append(p)
        if idx != len(panels)-1:
            composite_parts.append(sep)
    composite = np.concatenate(composite_parts, axis=1)
    return composite, labels


def add_labels_dynamic(img, labels, separator_width=10, font_scale=0.55, thickness=1):
    h, w = img.shape[:2]
    n = len(labels)
    # Number of separators = n-1
    total_sep = separator_width * (n - 1)
    panel_w = (w - total_sep) / n
    for i, label in enumerate(labels):
        # Panel center x
        cx = int(i * (panel_w + separator_width) + panel_w / 2)
        cv2.putText(img, label, (cx - 70, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0,0,255), thickness, lineType=cv2.LINE_AA)
    return img


def print_metadata_summary(metadata, entries_count, factors):
    print('Dataset metadata summary:')
    if not metadata:
        print('  (no metadata.json found)')
    else:
        keys = [
            'requested_images','selected_images','n_tiles_per_image','tile_size',
            'downsample_factor','downsample_factors','total_sr_tiles','total_lr_tiles'
        ]
        for k in keys:
            if k in metadata:
                print(f'  {k}: {metadata[k]}')
        if 'degradations' in metadata:
            print(f"  degradations: {metadata['degradations']}")
        if 'time' in metadata and 'duration_seconds' in metadata['time']:
            print(f"  generation_time_s: {metadata['time']['duration_seconds']}")
    print(f'  factors_detected: {factors}')
    print(f'  entries_available: {entries_count}')


def main():
    parser = argparse.ArgumentParser(description='Visualize multi-scale SR/LR tile sets')
    parser.add_argument('dataset_dir', type=str, help='Path to dataset root (with sr/ and lr_x*/ )')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle order of tiles')
    parser.add_argument('--seed', type=int, default=42, help='Seed for shuffling')
    parser.add_argument('--separator_width', type=int, default=10, help='Separator width between panels')
    parser.add_argument('--allow_partial', action='store_true', help='Include tiles even if some LR scales missing')
    args = parser.parse_args()

    entries, metadata, factors = load_image_entries(
        args.dataset_dir, shuffle=args.shuffle, seed=args.seed, require_all=not args.allow_partial
    )
    print_metadata_summary(metadata, len(entries), factors)

    if len(entries) == 0:
        print('No entries found!')
        return

    print('\nControls:')
    print('  a: Previous')
    print('  z: Next')
    print('  q/ESC: Quit')
    print('  r: Reset to first')
    if args.shuffle:
        print(f'  Order: shuffled (seed={args.seed})')
    else:
        print('  Order: sequential')
    if args.allow_partial:
        print('  Mode: partial (tiles with missing some scales included)')
    else:
        print('  Mode: strict (only tiles with all scales)')

    idx = 0
    total = len(entries)

    while True:
        sr_path, lr_paths_dict = entries[idx]
        try:
            composite, labels = create_composite(sr_path, lr_paths_dict, factors, separator_width=args.separator_width)
        except RuntimeError as e:
            print(f'Error for index {idx}: {e}')
            idx = (idx + 1) % total
            continue

        composite = add_labels_dynamic(composite, labels, separator_width=args.separator_width)
        counter = f"Tile {idx+1}/{total} - {Path(sr_path).name}".strip()
        cv2.putText(composite, counter, (10, composite.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255,255,255), 2)

        cv2.imshow('SR Multi-Scale Viewer', composite)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('z'):
            idx = (idx + 1) % total
        elif key == ord('a'):
            idx = (idx - 1) % total
        elif key == ord('r'):
            idx = 0

    cv2.destroyAllWindows()
    print('Viewer closed.')


if __name__ == '__main__':
    main()
