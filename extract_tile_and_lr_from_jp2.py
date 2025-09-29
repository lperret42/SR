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
from collections import defaultdict

MIN_SIZE_BYTES_DEFAULT = 1 * 1024 * 1024  # 1MB threshold


def sanitize_city_name(name):
    return name.replace(' ', '_')


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


def extract_center_grid_tiles(img, tile_size, n_tiles=None):
    """Extract a centered perfect square grid of non-overlapping tiles.
    If n_tiles is None, use the maximum centered square grid that fits."""
    h, w = img.shape[:2]
    max_side = min(h, w) // tile_size
    if max_side <= 0:
        raise ValueError(
            f"Image too small for tile_size={tile_size}: ({w}x{h})"
        )

    if n_tiles is None:
        side = max_side
    else:
        if n_tiles <= 0:
            raise ValueError("n_tiles must be > 0")
        side = int(math.isqrt(n_tiles))
        if side * side != n_tiles:
            raise ValueError(f"n_tiles={n_tiles} is not a perfect square")
        if side > max_side:
            raise ValueError(
                f"Cannot extract {n_tiles} tiles of {tile_size}px: required square {(side * tile_size)}px exceeds image ({w}x{h})"
            )

    n_tiles_effective = side * side
    region_size = side * tile_size
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
    assert len(tiles) == n_tiles_effective
    return tiles


def build_city_lookup(input_dir):
    """Read download_metadata.json if present to map JP2 paths to city folders."""
    manifest_path = Path(input_dir) / "download_metadata.json"
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Warning: could not parse {manifest_path}: {exc}")
        return {}

    cities = data.get("cities")
    if not isinstance(cities, list):
        return {}

    lookup = {}
    for city_entry in cities:
        city_name = city_entry.get("city")
        tiles = city_entry.get("tiles", [])
        for tile in tiles:
            files = tile.get("files", [])
            for rel_path in files:
                rel = Path(rel_path)
                if rel.suffix.lower() != ".jp2":
                    continue
                abs_path = (Path(input_dir) / rel).resolve()
                lookup[abs_path] = {
                    "city": city_name,
                    "city_folder": rel.parts[0] if rel.parts else sanitize_city_name(city_name or "unknown"),
                }
    return lookup


def filter_city(lookup_entry, allowed_cities):
    if not allowed_cities:
        return True
    if lookup_entry is None:
        return False
    return lookup_entry.get("city_folder") in allowed_cities or lookup_entry.get("city") in allowed_cities


def sample_jp2_images_by_size(
    input_dir,
    n_images,
    min_size_bytes,
    seed,
    city_lookup=None,
    allowed_cities=None,
    verbose=False,
):
    """Randomly sample up to n_images valid JP2 files from directory, skipping corrupted ones.
    Returns (selected_list, stats_dict)."""
    city_lookup = city_lookup or {}
    if verbose:
        print("Enumerating JP2 files...", flush=True)
    all_jp2 = [p.resolve() for p in Path(input_dir).rglob("*.jp2")]
    if not all_jp2:
        raise FileNotFoundError(f"No JP2 images found in {input_dir}")
    if verbose:
        print(f"  -> Found {len(all_jp2)} JP2 files", flush=True)

    per_city_total = defaultdict(int)
    per_city_kept = defaultdict(int)
    per_city_selected = defaultdict(int)
    candidates = []
    filtered_size = 0
    filtered_city = 0
    for p in all_jp2:
        lookup_entry = city_lookup.get(p)
        city_name = lookup_entry.get("city_folder") if lookup_entry else "_unknown"
        per_city_total[city_name] += 1
        if p.stat().st_size < min_size_bytes:
            filtered_size += 1
            continue
        if allowed_cities is not None and not filter_city(lookup_entry, allowed_cities):
            filtered_city += 1
            continue
        candidates.append((p, lookup_entry))
        per_city_kept[city_name] += 1

    if not candidates:
        raise RuntimeError("No JP2 images matching the criteria (size/city filters)")

    rnd = random.Random(seed)
    rnd.shuffle(candidates)

    target = n_images if n_images is not None else len(candidates)

    selected = []
    corrupted = 0
    decode_iter = tqdm(
        candidates,
        desc="Validating JP2",
        unit="file",
        disable=not verbose,
    )
    for path, lookup_entry in decode_iter:
        img = load_jp2_image(path)
        if img is None:
            corrupted += 1
            continue
        city_name = lookup_entry.get("city_folder") if lookup_entry else "_unknown"
        selected.append({
            "path": path,
            "city": lookup_entry.get("city") if lookup_entry else None,
            "city_folder": lookup_entry.get("city_folder") if lookup_entry else None,
        })
        per_city_selected[city_name] += 1
        if len(selected) >= target:
            break

    if not selected:
        raise RuntimeError("No valid JP2 images after filtering")

    if n_images is not None and len(selected) < n_images:
        raise RuntimeError(
            f"Only {len(selected)} valid images found (requested {n_images})."
        )
    stats = {
        "total_files": len(all_jp2),
        "kept_after_filters": len(candidates),
        "filtered_by_size": filtered_size,
        "filtered_by_city": filtered_city,
        "corrupted": corrupted,
        "requested": n_images if n_images is not None else "all",
        "per_city_total": dict(per_city_total),
        "per_city_kept": dict(per_city_kept),
        "per_city_selected": dict(per_city_selected),
    }
    if verbose:
        print(
            f"  -> Valid JP2 ready: {len(selected)} (corrupted skipped={corrupted})",
            flush=True,
        )
    return selected, stats


def build_subdirs(base_dir, city_folder, base_name):
    if city_folder:
        return base_dir / city_folder / base_name
    return base_dir / base_name


def process_single_image(img_path_str, group_label, output_root_str, downsample_factors, tile_size, n_tiles, blur_kernel, noise_std):
    """Worker function executed in a separate process. Returns a dict with results.
    downsample_factors: list[int]
    group_label: optional city folder name to use for sub-directories"""
    start = time.time()
    output_root = Path(output_root_str)
    img_path = Path(img_path_str)
    sr_dir = output_root / "sr"

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
            "group": group_label,
        }

    h, w = img.shape[:2]
    base_name = img_path.stem
    sr_sub = build_subdirs(sr_dir, group_label, base_name)
    sr_sub.mkdir(exist_ok=True, parents=True)

    lr_sub_dirs = {}
    for f in downsample_factors:
        lr_sub = build_subdirs(output_root / f"lr_x{f}", group_label, base_name)
        lr_sub.mkdir(exist_ok=True, parents=True)
        lr_sub_dirs[f] = lr_sub

    try:
        tiles = extract_center_grid_tiles(img, tile_size, n_tiles=n_tiles)
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
                "tiling_mode": "max_fit" if n_tiles is None else "fixed",
                "requested_n_tiles": n_tiles,
                "tiles": [],
                "group": group_label,
            },
            "duration": time.time() - start,
            "group": group_label,
        }

    per_image = {
        "filename": img_path.name,
        "width": w,
        "height": h,
        "tiles": [],
        "n_tiles": len(tiles),
        "tiling_mode": "max_fit" if n_tiles is None else "fixed",
        "requested_n_tiles": n_tiles,
        "processing_time_seconds": None,
        "group": group_label,
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
        "group": group_label,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract centered grid tiles from random JP2 images and generate LR counterparts (multiprocess, multi-scale)")
    parser.add_argument("input_dir", type=str, help="Directory containing JP2 images (recursively). Can be download_pcrs batch output root.")
    parser.add_argument("--n_images", type=int, default=None, help="Number of JP2 images to randomly sample (default: all valid)")
    parser.add_argument("--n_tiles", type=int, default=None, help="Number of tiles per image (perfect square). Default: max centered grid")
    parser.add_argument("--tile_size", type=int, default=256, help="Square tile size in pixels")
    parser.add_argument("--downsample", type=int, nargs="+", default=[4], help="One or more downsampling factors (e.g. --downsample 4 8)")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory")
    parser.add_argument("--blur_kernel", type=int, default=None, help="Gaussian blur kernel size (must be odd)")
    parser.add_argument("--noise_std", type=float, default=None, help="Gaussian noise std dev")
    parser.add_argument("--min_size_bytes", type=int, default=MIN_SIZE_BYTES_DEFAULT, help="Minimum JP2 file size to consider (default 1MB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: all CPUs)")
    parser.add_argument("--cities", nargs="+", default=None, help="Optional list of city folders (sanitized or original names) to include from download metadata")
    parser.add_argument("--reuse_output", action="store_true", help="Reuse existing output directory instead of deleting it (⚠️ may mix with previous run)")
    args = parser.parse_args()

    start_time = time.time()

    random.seed(args.seed)
    np.random.seed(args.seed)

    factors = []
    for f in args.downsample:
        if f not in factors:
            factors.append(f)
    for f in factors:
        if f < 2:
            raise ValueError(f"Downsample factor must be >=2, got {f}")
    multi_scale = len(factors) > 1

    if args.blur_kernel is not None and args.blur_kernel % 2 == 0:
        print("Warning: blur_kernel must be odd, adjusting ->", args.blur_kernel + 1)
        args.blur_kernel += 1

    if args.n_tiles is not None:
        if int(math.isqrt(args.n_tiles)) ** 2 != args.n_tiles:
            raise ValueError(f"n_tiles={args.n_tiles} is not a perfect square")

    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    city_lookup = build_city_lookup(input_path)
    allowed_cities = None
    if args.cities:
        allowed_cities = {sanitize_city_name(c) for c in args.cities}
        allowed_cities.update(args.cities)
        if not city_lookup:
            raise ValueError("City filtering requested but no download metadata found to resolve cities")

    print("\n=== Extraction configuration ===", flush=True)
    print(f"Input directory      : {input_path}")
    print(f"Output directory     : {args.output_dir}")
    print(f"Tile size / mode     : {args.tile_size}px | {'max_fit' if args.n_tiles is None else f'{args.n_tiles} tiles'}")
    print(f"Downsample factors   : {factors}")
    print(f"Blur kernel / noise  : {args.blur_kernel or 'none'} / {args.noise_std or 'none'}")
    if args.cities:
        print(f"City filter          : {', '.join(args.cities)}")
    else:
        print("City filter          : none")
    if args.reuse_output:
        print("Output reuse         : enabled (files may accumulate)")
    else:
        print("Output reuse         : disabled (clean run)")
    print("===============================\n", flush=True)

    output_path = Path(args.output_dir)
    if output_path.exists():
        if args.reuse_output:
            print(f"Output directory exists and will be reused: {output_path}", flush=True)
        else:
            print(
                f"Output directory already exists and will be removed for a clean extraction: {output_path}\n"
                "Use --reuse_output if you want to keep existing tiles (previous files may be overwritten or mixed).",
                flush=True,
            )
            print("Deleting existing output directory...", flush=True)
            shutil.rmtree(output_path)
            print("Output directory removed.", flush=True)
    sr_dir = output_path / "sr"
    sr_dir.mkdir(parents=True)
    for f in factors:
        (output_path / f"lr_x{f}").mkdir(parents=True)

    target_desc = "all" if args.n_images is None else str(args.n_images)
    print("Scanning JP2 files...", flush=True)
    print(
        f"Sampling {target_desc} JP2 image(s) (size >= {args.min_size_bytes} bytes) from {args.input_dir} with seed {args.seed} ...",
        flush=True,
    )
    selected_images, sampling_stats = sample_jp2_images_by_size(
        args.input_dir,
        args.n_images,
        args.min_size_bytes,
        args.seed,
        city_lookup=city_lookup,
        allowed_cities=allowed_cities,
        verbose=True,
    )
    print(
        "Sampling summary: "
        f"total={sampling_stats['total_files']} | "
        f"after filters={sampling_stats['kept_after_filters']} | "
        f"size filtered={sampling_stats['filtered_by_size']} | "
        f"city filtered={sampling_stats['filtered_by_city']} | "
        f"decode failures={sampling_stats['corrupted']}",
        flush=True,
    )
    if sampling_stats["per_city_kept"]:
        print("Per-city JP2 kept (after filters):", flush=True)
        for city, count in sorted(sampling_stats["per_city_kept"].items()):
            label = city if city != "_unknown" else "(unknown)"
            selected_count = sampling_stats["per_city_selected"].get(city, 0)
            print(f"  - {label}: candidates={count}, selected={selected_count}", flush=True)
    else:
        print("Per-city JP2 kept: not available (no metadata.json)", flush=True)

    print(f"Selected {len(selected_images)} image(s) for extraction.", flush=True)

    degradations = []
    if args.blur_kernel:
        degradations.append(f"blur={args.blur_kernel}")
    if args.noise_std:
        degradations.append(f"noise_std={args.noise_std:.2f}")

    mapping_rows = []
    num_workers = args.num_workers or os.cpu_count() or 1
    tiling_mode = "max_fit" if args.n_tiles is None else "fixed"
    metadata = {
        "requested_images": args.n_images if args.n_images is not None else "all",
        "selected_images": len(selected_images),
        "n_tiles_per_image": args.n_tiles if args.n_tiles is not None else "max_fit",
        "tiling_mode": tiling_mode,
        "tile_size": args.tile_size,
        "downsample_factors": factors,
        **({"downsample_factor": factors[0]} if len(factors) == 1 else {}),
        "degradations": {
            "blur_kernel": args.blur_kernel,
            "noise_std": args.noise_std,
        },
        "images": [],
        "total_sr_tiles": 0,
        "total_lr_tiles": 0,
        "total_lr_tiles_per_factor": {f"x{f}": 0 for f in factors},
        "output_dirs": {
            "sr": str(sr_dir.relative_to(output_path)),
            **{f"lr_x{f}": f"lr_x{f}" for f in factors},
        },
        "skipped_corrupted": 0,
        "skipped_geometry_error": 0,
        "script_options": {
            "tile_size": args.tile_size,
            "n_tiles": args.n_tiles,
            "tiling_mode": tiling_mode,
            "downsample_factors": factors,
            "min_size_bytes": args.min_size_bytes,
            "seed": args.seed,
            "num_workers": num_workers,
            "cities": args.cities,
            "reuse_output": args.reuse_output,
            "progress_unit": "tile" if args.n_tiles is not None else "image",
        },
        "throughput_tiles_per_sec": None,
        "group_summary": defaultdict(lambda: {"images": 0, "sr_tiles": 0}),
        "sampling_stats": sampling_stats,
    }

    progress_total = len(selected_images) * args.n_tiles if args.n_tiles is not None else len(selected_images)
    progress_unit = "tile" if args.n_tiles is not None else "image"
    progress_desc = "Processing tiles" if args.n_tiles is not None else "Processing images"
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

    pbar = tqdm(total=progress_total, desc=progress_desc, unit=progress_unit)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for item in selected_images:
            futures.append(
                executor.submit(worker_fn, str(item["path"]), item.get("city_folder"))
            )
        for future in as_completed(futures):
            result = future.result()
            status = result["status"]
            group_label = result.get("group") or "ungrouped"
            if args.n_tiles is None:
                progress_increment = 1
            else:
                progress_increment = result["tiles"] if status == "ok" else args.n_tiles
            if status == "ok":
                metadata["total_sr_tiles"] += result["tiles"]
                metadata["total_lr_tiles"] += result["lr_tiles"]
                for f in factors:
                    metadata["total_lr_tiles_per_factor"][f"x{f}"] += result["tiles"]
                metadata["images"].append(result["per_image"])
                metadata["group_summary"][group_label]["images"] += 1
                metadata["group_summary"][group_label]["sr_tiles"] += result["tiles"]
                mapping_rows.extend(result["mapping"])
            elif status == "corrupted":
                metadata["skipped_corrupted"] += 1
            elif status == "geometry_error":
                metadata["skipped_geometry_error"] += 1
            pbar.update(progress_increment)
    pbar.close()

    csv_path = output_path / "mapping.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        if multi_scale:
            header = ["sr_path"] + [f"lr_x{f}" for f in factors]
            writer.writerow(header)
            writer.writerows(mapping_rows)
        else:
            writer.writerow(["sr_path", "lr_path"])
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

    metadata["group_summary"] = {k: v for k, v in metadata["group_summary"].items()}

    json_path = output_path / "metadata.json"
    with json_path.open("w") as jf:
        json.dump(metadata, jf, indent=2)

    print("\n==== SUMMARY ====")
    print(f"Requested images: {metadata['requested_images']}")
    print(f"Processed images (successful): {len(metadata['images'])}")
    print(f"Corrupted skipped: {metadata['skipped_corrupted']}")
    print(f"Geometry errors skipped: {metadata['skipped_geometry_error']}")
    if args.n_tiles is None:
        print("Tiles per successful image: max_fit (varies per JP2)")
    else:
        print(f"Tiles per successful image (target): {args.n_tiles}")
    print(f"Tiling mode recorded in metadata: {metadata['tiling_mode']}")
    print(f"Total SR tiles: {metadata['total_sr_tiles']}")
    if multi_scale:
        print(f"Downsample factors: {factors}")
        for f in factors:
            print(f"  LR x{f} tiles: {metadata['total_lr_tiles_per_factor'].get('x'+str(f), 0)}")
        print(f"Total LR tiles (all scales): {metadata['total_lr_tiles']}")
    else:
        print(f"Downsample factor: {factors[0]}")
        print(f"Total LR tiles: {metadata['total_lr_tiles']}")
    print(f"Throughput (SR tiles/sec): {metadata['throughput_tiles_per_sec']}")
    if degradations:
        print("Degradations: " + ", ".join(degradations))
    else:
        print("Degradations: none")
    if metadata["group_summary"]:
        print("Group summary:")
        for group, stats in metadata["group_summary"].items():
            print(f"  {group}: images={stats['images']}, sr_tiles={stats['sr_tiles']}")
    print(f"Seed: {args.seed}")
    print(f"Workers used: {num_workers}")
    print(f"CSV mapping: {csv_path}")
    print(f"Metadata JSON: {json_path}")
    print("Included in metadata.json: multi-scale aware keys, per-group summary, legacy compatibility maintained.")
    print(f"Elapsed time: {metadata['time']['duration_seconds']}s")
    print(f"SR directory: {sr_dir}")
    for f in factors:
        print(f"LR x{f} directory: {output_path / ('lr_x'+str(f))}")


if __name__ == "__main__":
    main()
