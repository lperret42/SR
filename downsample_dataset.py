#!/usr/bin/env python3
"""Downsample (subsample) an existing SR dataset produced by extract_tile_and_lr_from_jp2.py.

Creates a new dataset root containing a random subset of source images (and all their tiles)
while preserving structure (sr/<base>/..., lr_x<factor>/<base>/...).

Selection controls (mutually exclusive):
  --frac FLOAT        Fraction of source images to keep (default 0.1)
  --n_images INT      Exact number of source images to keep

Reproducible via --seed (default 42). Multiprocessing copy with --num_workers.
Output directory name: <input_root>_<NSELECTED>
If it exists it is removed first.

metadata.json and mapping.csv are regenerated for the subset.
"""
import argparse
import random
import shutil
import json
import csv
import time
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import os


def parse_args():
    p = argparse.ArgumentParser(description="Subsample an existing SR dataset")
    p.add_argument('dataset_dir', type=str, help='Path to existing dataset root (with sr/, lr_x*/ and metadata.json)')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--frac', type=float, default=0.1, help='Fraction of source images to keep (0,1]')
    g.add_argument('--n_images', type=int, default=None, help='Exact number of source images to keep')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--num_workers', type=int, default=None, help='Worker processes (default: all CPUs)')
    p.add_argument('--dry_run', action='store_true', help='Show selection summary without copying')
    return p.parse_args()


def load_original_dataset(root: Path):
    meta_path = root / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'metadata.json not found in {root}')
    with meta_path.open('r') as f:
        metadata = json.load(f)
    # Infer LR directory
    if 'downsample_factor' in metadata:
        lr_dir = root / f"lr_x{metadata['downsample_factor']}"
    else:
        # fallback to first lr_x* dir
        lr_candidates = [d for d in root.glob('lr_x*') if d.is_dir()]
        if not lr_candidates:
            raise FileNotFoundError('No lr_x* directory found')
        lr_dir = lr_candidates[0]
    sr_dir = root / 'sr'
    if not sr_dir.exists():
        raise FileNotFoundError('sr directory missing')
    if not lr_dir.exists():
        raise FileNotFoundError(f'LR directory missing: {lr_dir}')
    return metadata, sr_dir, lr_dir


def list_image_bases(sr_dir: Path):
    # Each subdirectory in sr/ corresponds to one source image base
    return sorted([d.name for d in sr_dir.iterdir() if d.is_dir()])


def select_bases(bases, frac, n_images, seed):
    rnd = random.Random(seed)
    rnd.shuffle(bases)
    if n_images is not None:
        if n_images <= 0:
            raise ValueError('--n_images must be > 0')
        if n_images > len(bases):
            raise ValueError(f'Requested {n_images} images but only {len(bases)} available')
        selected = bases[:n_images]
    else:
        if not (0 < frac <= 1):
            raise ValueError('--frac must be in (0,1]')
        k = max(1, math.ceil(len(bases) * frac))
        selected = bases[:k]
    selected.sort()
    return selected


def load_mapping(root: Path):
    mapping_path = root / 'mapping.csv'
    if not mapping_path.exists():
        raise FileNotFoundError('mapping.csv missing')
    rows = []
    with mapping_path.open('r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            if len(r) >= 2:
                rows.append((r[0], r[1]))
    return rows


def filter_mapping_for_bases(mapping_rows, selected_bases):
    selected_set = set(selected_bases)
    filtr = []
    per_base_counts = {b: 0 for b in selected_bases}
    for sr_path, lr_path in mapping_rows:
        # sr path pattern: sr/<base>/file.png
        parts = Path(sr_path).parts
        if len(parts) >= 3 and parts[0] == 'sr':
            base = parts[1]
            if base in selected_set:
                filtr.append((sr_path, lr_path))
                per_base_counts[base] += 1
    return filtr, per_base_counts


def worker_copy(base, sr_dir, lr_dir, out_root, lr_dir_name):
    sr_src = sr_dir / base
    lr_src = lr_dir / base
    sr_dst = out_root / 'sr' / base
    lr_dst = out_root / lr_dir_name / base
    sr_dst.mkdir(parents=True, exist_ok=True)
    lr_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sr_src.glob('*.png'):
        shutil.copy2(f, sr_dst / f.name)
        copied += 1
    for f in lr_src.glob('*.png'):
        shutil.copy2(f, lr_dst / f.name)
    return base, copied


def main():
    args = parse_args()
    start = time.time()
    input_root = Path(args.dataset_dir).resolve()

    metadata, sr_dir, lr_dir = load_original_dataset(input_root)
    lr_dir_name = lr_dir.name
    bases = list_image_bases(sr_dir)
    selected_bases = select_bases(bases, args.frac, args.n_images, args.seed)

    print(f'Found {len(bases)} source images (bases). Selecting {len(selected_bases)}.')

    mapping_rows = load_mapping(input_root)
    filtered_mapping, per_base_counts = filter_mapping_for_bases(mapping_rows, selected_bases)
    total_tiles = sum(per_base_counts.values())
    print(f'Selected tiles total (SR/LR pairs): {total_tiles}')

    out_root = input_root.parent / f"{input_root.name}_{len(selected_bases)}"
    if out_root.exists():
        print(f'Removing existing output directory: {out_root}')
        shutil.rmtree(out_root)
    if args.dry_run:
        print('Dry run complete. No data copied.')
        return
    (out_root / 'sr').mkdir(parents=True)
    (out_root / lr_dir_name).mkdir(parents=True)

    num_workers = args.num_workers or os.cpu_count() or 1
    print(f'Copying data with {num_workers} worker(s)...')

    worker_fn = worker_copy
    progress = tqdm(total=len(selected_bases), desc='Copying bases', unit='base')
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(worker_fn, b, sr_dir, lr_dir, out_root, lr_dir_name) for b in selected_bases]
        for fut in as_completed(futs):
            base, copied = fut.result()
            progress.update(1)
    progress.close()

    # Write new mapping.csv (paths already relative to original root, must adapt if root names differ?)
    # Because structure is identical under new root, we can reuse same relative paths.
    new_mapping_path = out_root / 'mapping.csv'
    with new_mapping_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sr_path', 'lr_path'])
        for row in filtered_mapping:
            w.writerow(row)

    # Build new metadata
    new_meta = {
        'parent_dataset_root': str(input_root),
        'subset_of': input_root.name,
        'requested_images': len(selected_bases),
        'selected_images': len(selected_bases),
        'n_tiles_per_image': metadata.get('n_tiles_per_image'),
        'tile_size': metadata.get('tile_size'),
        'downsample_factor': metadata.get('downsample_factor'),
        'degradations': metadata.get('degradations'),
        'total_sr_tiles': total_tiles,
        'total_lr_tiles': total_tiles,
        'output_dirs': {
            'sr': 'sr',
            'lr': lr_dir_name,
        },
        'script_options': {
            'selection_mode': 'n_images' if args.n_images is not None else 'frac',
            'frac': args.frac if args.n_images is None else None,
            'n_images': args.n_images,
            'seed': args.seed,
            'num_workers': num_workers,
        },
        'time': {},
        'images': [],
    }

    # Filter original per-image metadata if available
    orig_images = { (img.get('filename','').split('.')[0]): img for img in metadata.get('images', []) }
    for base in selected_bases:
        # Attempt match by base; copy and prune tiles list to those present (all tiles anyway)
        orig = orig_images.get(base)
        if orig:
            # Optionally filter tiles to ensure only those in mapping (they all are) -> keep
            new_meta['images'].append(orig)
        else:
            new_meta['images'].append({'filename': base, 'n_tiles': per_base_counts[base]})

    end = time.time()
    new_meta['time'] = {
        'start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start)),
        'end': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end)),
        'duration_seconds': round(end - start, 3),
    }
    if total_tiles > 0:
        new_meta['throughput_tiles_per_sec'] = round(total_tiles / (end - start), 3) if (end - start) > 0 else None

    with (out_root / 'metadata.json').open('w') as f:
        json.dump(new_meta, f, indent=2)

    print('\n==== SUMMARY ====')
    print(f'Input dataset: {input_root}')
    print(f'Output dataset: {out_root}')
    print(f'Selected bases: {len(selected_bases)} / {len(bases)}')
    print(f'Total tiles copied: {total_tiles}')
    print(f'Mapping file: {new_mapping_path}')
    print('Metadata keys:', ', '.join(sorted(new_meta.keys())))
    print(f'Duration: {new_meta["time"]["duration_seconds"]}s')
    print('Done.')


if __name__ == '__main__':
    main()
