#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import random
from tqdm import tqdm
import math
import shutil
import time
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

MIN_SIZE_BYTES_DEFAULT = 1 * 1024 * 1024  # 1MB threshold


def load_jp2_image(jp2_path):
    """Load JP2 image using OpenCV. Returns None if load fails."""
    img = cv2.imread(str(jp2_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None
    return img


def downsample_image(img, factor, add_blur=False, blur_kernel=None, add_noise=False, noise_std=None):
    """Downsample image with optional degradations (blur + noise)."""
    if add_blur and blur_kernel is not None:
        img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    h, w = img.shape[:2]
    new_h, new_w = h // factor, w // factor
    downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if add_noise and noise_std is not None:
        noise = np.random.normal(0, noise_std, downsampled.shape).astype(np.float32)
        downsampled = downsampled.astype(np.float32) + noise
        downsampled = np.clip(downsampled, 0, 255).astype(np.uint8)
    return downsampled


def extract_center_grid_tiles(img, n_tiles, tile_size):
    """Extract a centered perfect square grid of non-overlapping tiles.
    Returns list of (tile, x, y). Raises ValueError if impossible."""
    if n_tiles <= 0:
        raise ValueError("n_tiles must be > 0")
    side = int(math.isqrt(n_tiles))
    if side * side != n_tiles:
        raise ValueError(f"n_tiles={n_tiles} is not a perfect square")
    h, w = img.shape[:2]
    region_size = side * tile_size
    if region_size > h or region_size > w:
        raise ValueError(
            f"Cannot extract {n_tiles} tiles of {tile_size}px: required square {region_size}px exceeds image ({w}x{h})"
        )
    start_y = (h - region_size) // 2
    start_x = (w - region_size) // 2
    tiles = []
    for gy in range(side):
        for gx in range(side):
            y = start_y + gy * tile_size
            x = start_x + gx * tile_size
            tile = img[y:y+tile_size, x:x+tile_size]
            if tile.shape[:2] != (tile_size, tile_size):
                raise ValueError("Internal error: tile has wrong shape")
            tiles.append((tile, x, y))
    return tiles


def sample_jp2_images_by_size(input_dir, n_images, min_size_bytes, seed):
    """Sample JP2 images purely by file size threshold (>= min_size_bytes).
    No decoding performed here. Deterministic via provided seed."""
    all_jp2 = list(Path(input_dir).rglob("*.jp2"))
    if not all_jp2:
        raise FileNotFoundError(f"No JP2 images found in {input_dir}")
    candidates = [p for p in all_jp2 if p.stat().st_size >= min_size_bytes]
    if not candidates:
        raise RuntimeError(
            f"No JP2 images >= {min_size_bytes} bytes (found {len(all_jp2)} smaller files)"
        )
    rnd = random.Random(seed)
    rnd.shuffle(candidates)
    if len(candidates) < n_images:
        raise RuntimeError(
            f"Only {len(candidates)} images >= {min_size_bytes} bytes (requested {n_images})."
        )
    return candidates[:n_images]


def process_single_image(img_path_str, output_root_str, downsample_factors, tile_size, n_tiles, blur_kernel, noise_std):
    """Worker function executed in a separate process. Returns a dict with results.
    downsample_factors: list[int]"""
    start = time.time()
    output_root = Path(output_root_str)
    img_path = Path(img_path_str)
    sr_dir = output_root / "sr"

    # Ensure LR directories exist (avoid race with parents=True)
    for f in downsample_factors:
        (output_root / f"lr_x{f}").mkdir(exist_ok=True, parents=True)

    img = load_jp2_image(img_path)
    if img is None:
        return {
            "status": "corrupted",
            "filename": img_path.name,
            "tiles": 0,
            "lr_tiles": 0,
            "mapping": [],
            "per_image": None,
            "duration": time.time() - start,
        }

    h, w = img.shape[:2]
    base_name = img_path.stem
    sr_sub = sr_dir / base_name
    sr_sub.mkdir(exist_ok=True, parents=True)
    lr_sub_dirs = {f: (output_root / f"lr_x{f}" / base_name) for f in downsample_factors}
    for d in lr_sub_dirs.values():
        d.mkdir(exist_ok=True, parents=True)

    try:
        tiles = extract_center_grid_tiles(img, n_tiles, tile_size)
    except ValueError:
        return {
            "status": "geometry_error",
            "filename": img_path.name,
            "tiles": 0,
            "lr_tiles": 0,
            "mapping": [],
            "per_image": {
                "filename": img_path.name,
                "width": w,
                "height": h,
                "n_tiles": 0,
                "tiles": [],
            },
            "duration": time.time() - start,
        }

    per_image = {
        "filename": img_path.name,
        "width": w,
        "height": h,
        "tiles": [],
        "n_tiles": len(tiles),
        "processing_time_seconds": None,
    }
    mapping_rows = []

    multi = len(downsample_factors) > 1

    for i, (tile_hr, x, y) in enumerate(tiles):
        tile_hr_pil = Image.fromarray(tile_hr)
        sr_filename = sr_sub / f"{base_name}_tile_{i:04d}_x{x}_y{y}.png"
        tile_hr_pil.save(sr_filename)
        sr_rel = sr_filename.relative_to(output_root)

        lr_paths = {}
        for fct in downsample_factors:
            tile_lr = downsample_image(
                tile_hr,
                fct,
                add_blur=(blur_kernel is not None),
                blur_kernel=blur_kernel,
                add_noise=(noise_std is not None),
                noise_std=noise_std,
            )
            lr_filename = lr_sub_dirs[fct] / f"{base_name}_tile_{i:04d}_x{x}_y{y}.png"
            Image.fromarray(tile_lr).save(lr_filename)
            lr_rel = lr_filename.relative_to(output_root)
            lr_paths[f"x{fct}"] = str(lr_rel)

        if multi:
            row = [str(sr_rel)] + [lr_paths[f"x{f}"] for f in downsample_factors]
            mapping_rows.append(row)
            per_image["tiles"].append({
                "index": i,
                "hr_path": str(sr_rel),
                "lr_paths": lr_paths,
                "x": x,
                "y": y,
            })
        else:
            # single factor backward compat
            only_lr = lr_paths[f"x{downsample_factors[0]}"]
            mapping_rows.append((str(sr_rel), only_lr))
            per_image["tiles"].append({
                "index": i,
                "hr_path": str(sr_rel),
                "lr_path": only_lr,
                "x": x,
                "y": y,
            })

    per_image["processing_time_seconds"] = round(time.time() - start, 3)
    return {
        "status": "ok",
        "filename": img_path.name,
        "tiles": len(tiles),
        "lr_tiles": len(tiles) * len(downsample_factors),
        "mapping": mapping_rows,
        "per_image": per_image,
        "duration": per_image["processing_time_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract centered grid tiles from random JP2 images and generate LR counterparts (multiprocess, multi-scale)")
    parser.add_argument("input_dir", type=str, help="Directory containing JP2 images")
    parser.add_argument("--n_images", type=int, default=1, help="Number of JP2 images to randomly sample")
    parser.add_argument("--n_tiles", type=int, default=16, help="Number of tiles per image (perfect square)")
    parser.add_argument("--tile_size", type=int, default=256, help="Square tile size in pixels")
    # Allow one or multiple factors (default single 4 for backward compatibility)
    parser.add_argument("--downsample", type=int, nargs="+", default=[4], help="One or more downsampling factors (e.g. --downsample 4 8)")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory")
    parser.add_argument("--blur_kernel", type=int, default=None, help="Gaussian blur kernel size (must be odd)")
    parser.add_argument("--noise_std", type=float, default=None, help="Gaussian noise std dev")
    parser.add_argument("--min_size_bytes", type=int, default=MIN_SIZE_BYTES_DEFAULT, help="Minimum JP2 file size to consider (default 1MB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: all CPUs)")
    args = parser.parse_args()

    start_time = time.time()

    # Seed RNGs
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Normalize & validate downsample factors (preserve order, uniq)
    factors = []
    for f in args.downsample:
        if f not in factors:
            factors.append(f)
    for f in factors:
        if f < 2:
            raise ValueError(f"Downsample factor must be >=2, got {f}")
    multi_scale = len(factors) > 1

    # Validate blur kernel
    if args.blur_kernel is not None and args.blur_kernel % 2 == 0:
        print("Warning: blur_kernel must be odd, adjusting ->", args.blur_kernel + 1)
        args.blur_kernel += 1

    # Validate n_tiles is perfect square
    if int(math.isqrt(args.n_tiles)) ** 2 != args.n_tiles:
        raise ValueError(f"n_tiles={args.n_tiles} is not a perfect square")

    output_path = Path(args.output_dir)
    if output_path.exists():
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    sr_dir = output_path / "sr"
    sr_dir.mkdir(parents=True)
    for f in factors:
        (output_path / f"lr_x{f}").mkdir(parents=True)

    print(
        f"Sampling {args.n_images} JP2 image(s) (size >= {args.min_size_bytes} bytes) from {args.input_dir} with seed {args.seed} ..."
    )
    selected_images = sample_jp2_images_by_size(
        args.input_dir, args.n_images, args.min_size_bytes, args.seed
    )
    print(f"Selected {len(selected_images)} image(s).")

    degradations = []
    if args.blur_kernel:
        degradations.append(f"blur={args.blur_kernel}")
    if args.noise_std:
        degradations.append(f"noise_std={args.noise_std:.2f}")

    mapping_rows = []
    metadata = {
        "requested_images": args.n_images,
        "selected_images": len(selected_images),
        "n_tiles_per_image": args.n_tiles,
        "tile_size": args.tile_size,
        "downsample_factors": factors,
        # keep single-factor key for backward compatibility
        **({"downsample_factor": factors[0]} if len(factors) == 1 else {}),
        "degradations": {
            "blur_kernel": args.blur_kernel,
            "noise_std": args.noise_std,
        },
        "images": [],
        "total_sr_tiles": 0,
        "total_lr_tiles": 0,  # combined across factors
        "total_lr_tiles_per_factor": {},
        "output_dirs": {
            "sr": str(sr_dir.relative_to(output_path)),
            **{f"lr_x{f}": f"lr_x{f}" for f in factors},
        },
        "skipped_corrupted": 0,
        "skipped_geometry_error": 0,
        "script_options": {
            "tile_size": args.tile_size,
            "n_tiles": args.n_tiles,
            "downsample_factors": factors,
            "min_size_bytes": args.min_size_bytes,
            "seed": args.seed,
            "num_workers": args.num_workers or os.cpu_count(),
        },
        "throughput_tiles_per_sec": None,
    }

    total_expected_tiles = len(selected_images) * args.n_tiles
    num_workers = args.num_workers or os.cpu_count() or 1
    print(f"Using {num_workers} worker process(es)...")

    worker_fn = partial(
        process_single_image,
        output_root_str=str(output_path),
        downsample_factors=factors,
        tile_size=args.tile_size,
        n_tiles=args.n_tiles,
        blur_kernel=args.blur_kernel,
        noise_std=args.noise_std,
    )

    pbar = tqdm(total=total_expected_tiles, desc="Processing tiles", unit="tile")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(worker_fn, str(p)): p for p in selected_images}
        for future in as_completed(future_to_path):
            result = future.result()
            status = result["status"]
            if status == "ok":
                metadata["total_sr_tiles"] += result["tiles"]
                metadata["total_lr_tiles"] += result["lr_tiles"]
                # per factor counts (each factor adds same number of tiles)
                for f in factors:
                    metadata["total_lr_tiles_per_factor"].setdefault(f"x{f}", 0)
                    metadata["total_lr_tiles_per_factor"][f"x{f}"] += result["tiles"]
                metadata["images"].append(result["per_image"])
                mapping_rows.extend(result["mapping"])
                pbar.update(result["tiles"])
            elif status == "corrupted":
                metadata["skipped_corrupted"] += 1
            elif status == "geometry_error":
                metadata["skipped_geometry_error"] += 1
    pbar.close()

    # Write CSV mapping
    csv_path = output_path / "mapping.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        if multi_scale:
            header = ["sr_path"] + [f"lr_x{f}" for f in factors]
            writer.writerow(header)
            writer.writerows(mapping_rows)
        else:
            writer.writerow(["sr_path", "lr_path"])  # backward compat
            writer.writerows(mapping_rows)

    end_time = time.time()
    duration = end_time - start_time
    metadata["time"] = {
        "start": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start_time)),
        "end": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end_time)),
        "duration_seconds": round(duration, 3),
    }
    if metadata["total_sr_tiles"] > 0 and duration > 0:
        metadata["throughput_tiles_per_sec"] = round(metadata["total_sr_tiles"] / duration, 3)

    json_path = output_path / "metadata.json"
    with json_path.open("w") as jf:
        json.dump(metadata, jf, indent=2)

    # Final summary
    print("\n==== SUMMARY ====")
    print(f"Requested images: {metadata['requested_images']}")
    print(f"Processed images (successful): {len(metadata['images'])}")
    print(f"Corrupted skipped: {metadata['skipped_corrupted']}")
    print(f"Geometry errors skipped: {metadata['skipped_geometry_error']}")
    print(f"Tiles per successful image (target): {args.n_tiles}")
    print(f"Total SR tiles: {metadata['total_sr_tiles']}")
    if multi_scale:
        print(f"Downsample factors: {factors}")
        for f in factors:
            print(f"  LR x{f} tiles: {metadata['total_lr_tiles_per_factor']['x'+str(f)]}")
        print(f"Total LR tiles (all scales): {metadata['total_lr_tiles']}")
    else:
        print(f"Downsample factor: {factors[0]}")
        print(f"Total LR tiles: {metadata['total_lr_tiles']}")
    print(f"Throughput (SR tiles/sec): {metadata['throughput_tiles_per_sec']}")
    if degradations:
        print("Degradations: " + ", ".join(degradations))
    else:
        print("Degradations: none")
    print(f"Seed: {args.seed}")
    print(f"Workers used: {num_workers}")
    print(f"CSV mapping: {csv_path}")
    print(f"Metadata JSON: {json_path}")
    print("Included in metadata.json: multi-scale aware keys (downsample_factors, total_lr_tiles_per_factor) plus legacy compatibility keys when single factor.")
    print(f"Elapsed time: {metadata['time']['duration_seconds']}s")
    print(f"SR directory: {sr_dir}")
    for f in factors:
        print(f"LR x{f} directory: {output_path / ('lr_x'+str(f))}")


if __name__ == "__main__":
    main()
