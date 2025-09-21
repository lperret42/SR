#!/usr/bin/env python3
"""Merge two SR (super-resolution) datasets produced by extract_tile_and_lr_from_jp2.py.

Supports single-scale (legacy) and multi-scale (multiple lr_x<factor> dirs) datasets.
Preserves structure and recomputes combined mapping.csv and metadata.json.

Compatibility checks (must match between both datasets):
  - tile_size
  - n_tiles_per_image
  - downsample_factors (order ignored; set equality required)
  - presence/absence of degradations parameters (values can differ; both recorded)

Duplicate base directory handling (base = subdirectory inside sr/):
  --on-duplicate error|skip|rename (default: error)
    * error  : raise if any common base names
    * skip   : keep bases from first dataset; discard duplicates from second
    * rename : rename duplicates from second dataset with a suffix (default '_B')

If --output_dir is not provided, an automatic name is generated:
  <datasetA_name>__MERGED__<datasetB_name>_<TOTAL_BASES>

Parallel copy controlled by --num_workers (default: all CPUs).

Result metadata additions:
  - merged_from: [nameA, nameB]
  - parent_datasets: [abs_path_A, abs_path_B]
  - merge_strategy_duplicates
  - per-dataset degradations recorded under `source_degradations`
  - when duplicates renamed, original base stored in image entry under `original_base`

mapping.csv format:
  - Single factor: sr_path,lr_path (legacy) if only one factor overall
  - Multi-scale  : sr_path,lr_x<factor1>,lr_x<factor2>, ... (factors sorted ascending)

Paths inside mapping & metadata remain relative to new root.
"""
import argparse
import json
import csv
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# -------------------- Helpers to load datasets --------------------

def load_metadata(root: Path):
    meta_path = root / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {root}")
    with meta_path.open('r') as f:
        return json.load(f)


def detect_factors(meta):
    if 'downsample_factors' in meta and meta['downsample_factors']:
        return list(meta['downsample_factors'])
    if 'downsample_factor' in meta:
        return [meta['downsample_factor']]
    # Fallback: try infer from output_dirs
    if 'output_dirs' in meta:
        facs = []
        for k in meta['output_dirs'].keys():
            if k.startswith('lr_x'):
                try:
                    facs.append(int(k.split('lr_x')[1]))
                except Exception:
                    pass
        if facs:
            return sorted(set(facs))
    raise ValueError('Could not determine downsample factors from metadata')


def read_mapping(root: Path):
    mpath = root / 'mapping.csv'
    if not mpath.exists():
        raise FileNotFoundError(f"mapping.csv missing in {root}")
    with mpath.open('r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    return header, rows


def collect_bases(sr_root: Path):
    return sorted([d.name for d in sr_root.iterdir() if d.is_dir()])


def normalize_mapping_rows(header, rows, factors):
    """Return list of dict entries: { 'sr_path': str, factor: lr_path, ... }.
    Handles legacy two-column format.
    """
    norm = []
    if len(header) == 2 and header == ['sr_path', 'lr_path'] and len(factors) == 1:
        f = factors[0]
        for r in rows:
            if len(r) < 2:
                continue
            norm.append({'sr_path': r[0], f: r[1]})
    else:
        # Expect sr_path + lr_x<factor> columns
        # Build map from column names to factor
        factor_cols = {}
        for col in header[1:]:
            if col.startswith('lr_x'):
                try:
                    factor = int(col.split('lr_x')[1])
                except Exception:
                    continue
                factor_cols[col] = factor
        for r in rows:
            if len(r) != len(header):
                continue
            entry = {'sr_path': r[0]}
            for idx, col in enumerate(header[1:], start=1):
                if col in factor_cols:
                    entry[factor_cols[col]] = r[idx]
            norm.append(entry)
    return norm


def rewrite_paths_for_base(entry, old_base, new_base, factors):
    """Update mapping entry when a base is renamed."""
    if old_base == new_base:
        return entry
    parts = Path(entry['sr_path']).parts
    if len(parts) >= 3 and parts[0] == 'sr' and parts[1] == old_base:
        entry['sr_path'] = str(Path('sr') / new_base / parts[2])
    for f in factors:
        if f in entry:
            p = Path(entry[f])
            p_parts = p.parts
            # Expect lr_x<factor>/<base>/<file>
            if len(p_parts) >= 3 and p_parts[0] == f'lr_x{f}' and p_parts[1] == old_base:
                entry[f] = str(Path(p_parts[0]) / new_base / p_parts[2])
    return entry


def adjust_image_metadata(image_entry, old_base, new_base, factors, multi_scale):
    if old_base == new_base:
        return image_entry
    # Deep-ish update of tile paths
    for tile in image_entry.get('tiles', []):
        # hr path
        if 'hr_path' in tile:
            p = Path(tile['hr_path'])
            parts = p.parts
            if len(parts) >= 3 and parts[0] == 'sr' and parts[1] == old_base:
                tile['hr_path'] = str(Path('sr') / new_base / parts[2])
        # single scale
        if not multi_scale and 'lr_path' in tile:
            p = Path(tile['lr_path'])
            parts = p.parts
            if len(parts) >= 3 and parts[1] == old_base:
                tile['lr_path'] = str(Path(parts[0]) / new_base / parts[2])
        # multi-scale lr_paths dict
        if multi_scale and 'lr_paths' in tile:
            for key, v in tile['lr_paths'].items():
                p = Path(v)
                parts = p.parts
                # lr_xF/<base>/<file>
                if len(parts) >= 3 and parts[1] == old_base:
                    tile['lr_paths'][key] = str(Path(parts[0]) / new_base / parts[2])
    image_entry['original_base'] = old_base
    return image_entry

# -------------------- Copy worker --------------------

def copy_base(dataset_root: str, base: str, out_root: str, factors, new_base: str):
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)
    sr_src = dataset_root / 'sr' / base
    sr_dst = out_root / 'sr' / new_base
    sr_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sr_src.glob('*.png'):
        shutil.copy2(f, sr_dst / f.name)
        copied += 1
    for factor in factors:
        lr_src = dataset_root / f'lr_x{factor}' / base
        lr_dst = out_root / f'lr_x{factor}' / new_base
        lr_dst.mkdir(parents=True, exist_ok=True)
        for f in lr_src.glob('*.png'):
            shutil.copy2(f, lr_dst / f.name)
    return base, new_base, copied

# -------------------- Main merge logic --------------------

def parse_args():
    ap = argparse.ArgumentParser(description='Merge two SR datasets (single or multi-scale)')
    ap.add_argument('dataset_a', type=str, help='First dataset root')
    ap.add_argument('dataset_b', type=str, help='Second dataset root')
    ap.add_argument('--output_dir', type=str, default=None, help='Output directory (auto if omitted)')
    ap.add_argument('--on-duplicate', type=str, default='error', choices=['error','skip','rename'], help='Strategy for duplicate base names')
    ap.add_argument('--rename-suffix', type=str, default='_B', help='Suffix when renaming duplicate bases (used if --on-duplicate rename)')
    ap.add_argument('--num_workers', type=int, default=None, help='Number of worker processes (default: all CPUs)')
    return ap.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    root_a = Path(args.dataset_a).resolve()
    root_b = Path(args.dataset_b).resolve()

    meta_a = load_metadata(root_a)
    meta_b = load_metadata(root_b)

    factors_a = detect_factors(meta_a)
    factors_b = detect_factors(meta_b)

    # Normalize factor sets & validate
    if set(factors_a) != set(factors_b):
        raise ValueError(f"Incompatible factors: {factors_a} vs {factors_b}")
    # Use sorted ascending final order
    factors = sorted(set(factors_a))
    multi_scale = len(factors) > 1

    # Validate shared properties
    for key in ['tile_size','n_tiles_per_image']:
        va = meta_a.get(key)
        vb = meta_b.get(key)
        if va != vb:
            raise ValueError(f"Incompatible {key}: {va} vs {vb}")

    # Read mappings
    header_a, rows_a = read_mapping(root_a)
    header_b, rows_b = read_mapping(root_b)
    norm_a = normalize_mapping_rows(header_a, rows_a, factors)
    norm_b = normalize_mapping_rows(header_b, rows_b, factors)

    # Bases
    bases_a = collect_bases(root_a / 'sr')
    bases_b = collect_bases(root_b / 'sr')
    dup = sorted(set(bases_a) & set(bases_b))
    base_map_b = {}  # old_base -> new_base
    if dup:
        if args.on_duplicate == 'error':
            raise ValueError(f"Duplicate bases found: {dup}")
        elif args.on_duplicate == 'skip':
            for b in dup:
                print(f"Skipping duplicate base from B: {b}")
            bases_b = [b for b in bases_b if b not in dup]
        elif args.on_duplicate == 'rename':
            for b in dup:
                new_base = b + args.rename_suffix
                # ensure uniqueness
                counter = 1
                while new_base in bases_a or new_base in bases_b or new_base in base_map_b.values():
                    new_base = f"{b}{args.rename_suffix}{counter}"
                    counter += 1
                base_map_b[b] = new_base
                print(f"Renaming duplicate base {b} -> {new_base}")
    # Map unchanged bases
    for b in bases_b:
        if b not in base_map_b:
            base_map_b[b] = b

    # Prepare output dir
    if args.output_dir:
        out_root = Path(args.output_dir).resolve()
    else:
        out_root = Path(f"{root_a.name}__MERGED__{root_b.name}")
    if out_root.exists():
        print(f"Removing existing output directory: {out_root}")
        shutil.rmtree(out_root)
    (out_root / 'sr').mkdir(parents=True)
    for f in factors:
        (out_root / f"lr_x{f}").mkdir(parents=True)

    # Build mapping rows combined
    combined_mapping_entries = []
    # Dataset A entries
    for e in norm_a:
        combined_mapping_entries.append(e.copy())
    for e in norm_b:
        # rewrite if base renamed
        sr_parts = Path(e['sr_path']).parts
        if len(sr_parts) >= 3:
            old_base = sr_parts[1]
            new_base = base_map_b.get(old_base, old_base)
            if new_base != old_base:
                e = rewrite_paths_for_base(e, old_base, new_base, factors)
        combined_mapping_entries.append(e.copy())

    # Build final mapping.csv header & rows
    if multi_scale:
        mapping_header = ['sr_path'] + [f"lr_x{f}" for f in factors]
        mapping_rows_out = []
        for e in combined_mapping_entries:
            row = [e['sr_path']] + [e.get(f, '') for f in factors]
            mapping_rows_out.append(row)
    else:
        fct = factors[0]
        mapping_header = ['sr_path','lr_path']
        mapping_rows_out = []
        for e in combined_mapping_entries:
            mapping_rows_out.append([e['sr_path'], e.get(fct,'')])

    # Combine metadata images
    images = []
    # Index meta_b images by base (stem of filename before '.') if available
    meta_a_images = meta_a.get('images', [])
    meta_b_images = meta_b.get('images', [])

    # Helper to extract base from a tile path (hr_path)
    def base_from_tile(tile):
        p = Path(tile.get('hr_path',''))
        parts = p.parts
        if len(parts) >= 3 and parts[0] == 'sr':
            return parts[1]
        return None

    # Add images from A directly
    for im in meta_a_images:
        images.append(im)

    # Add images from B with possible rename adjustments
    for im in meta_b_images:
        # Determine base from first tile if rename needed
        old_base = None
        if im.get('tiles'):
            old_base = base_from_tile(im['tiles'][0])
        if old_base and old_base in base_map_b and base_map_b[old_base] != old_base:
            new_base = base_map_b[old_base]
            adjust_image_metadata(im, old_base, new_base, factors, multi_scale)
        images.append(im)

    # Aggregate counts
    total_sr_tiles = (meta_a.get('total_sr_tiles',0) + meta_b.get('total_sr_tiles',0))
    if multi_scale:
        # Sum per factor; assume each dataset has per-factor counts or infer
        per_factor = {f:0 for f in factors}
        per_a = meta_a.get('total_lr_tiles_per_factor')
        per_b = meta_b.get('total_lr_tiles_per_factor')
        if per_a and per_b:
            for f in factors:
                per_factor[f] = per_a.get(f'x{f}',0) + per_b.get(f'x{f}',0)
        else:
            # Fallback assume equality to SR tiles
            for f in factors:
                per_factor[f] = total_sr_tiles
        total_lr_tiles = sum(per_factor.values())
    else:
        per_factor = {factors[0]: total_sr_tiles}
        total_lr_tiles = total_sr_tiles

    # Build new metadata
    merged_meta = {
        'merged_from': [root_a.name, root_b.name],
        'parent_datasets': [str(root_a), str(root_b)],
        'merge_strategy_duplicates': args.on_duplicate,
        'requested_images': meta_a.get('requested_images',0) + meta_b.get('requested_images',0),
        'selected_images': meta_a.get('selected_images',0) + meta_b.get('selected_images',0),
        'n_tiles_per_image': meta_a.get('n_tiles_per_image'),
        'tile_size': meta_a.get('tile_size'),
        'downsample_factors': factors,
        **({'downsample_factor': factors[0]} if not multi_scale else {}),
        'degradations': meta_a.get('degradations'),  # primary
        'source_degradations': {
            root_a.name: meta_a.get('degradations'),
            root_b.name: meta_b.get('degradations'),
        },
        'images': images,
        'total_sr_tiles': total_sr_tiles,
        'total_lr_tiles': total_lr_tiles,
        'total_lr_tiles_per_factor': {f'x{f}': per_factor[f] for f in factors},
        'output_dirs': {
            'sr': 'sr',
            **{f'lr_x{f}': f'lr_x{f}' for f in factors}
        },
        'script_options': {
            'merge_on_duplicate': args.on_duplicate,
            'rename_suffix': args.rename_suffix if args.on_duplicate=='rename' else None,
            'num_workers': args.num_workers or os.cpu_count(),
        },
        'skipped_corrupted': (meta_a.get('skipped_corrupted',0) + meta_b.get('skipped_corrupted',0)),
        'skipped_geometry_error': (meta_a.get('skipped_geometry_error',0) + meta_b.get('skipped_geometry_error',0)),
        'time': {},
    }

    # Copy files in parallel
    tasks = []
    for b in bases_a:
        tasks.append( (root_a, b, b) )
    for b in bases_b:
        tasks.append( (root_b, b, base_map_b[b]) )

    num_workers = args.num_workers or os.cpu_count() or 1
    from tqdm import tqdm
    pbar = tqdm(total=len(tasks), desc='Copying bases', unit='base')

    start_copy = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for ds_root, base_old, base_new in tasks:
            futures.append(ex.submit(copy_base, str(ds_root), base_old, str(out_root), factors, base_new))
        for fut in as_completed(futures):
            _ = fut.result()
            pbar.update(1)
    pbar.close()
    end_copy = time.time()

    merged_meta['time'] = {
        'start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start_time)),
        'end': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end_copy)),
        'duration_seconds': round(end_copy - start_time, 3),
        'copy_duration_seconds': round(end_copy - start_copy, 3),
    }
    if total_sr_tiles > 0 and (end_copy - start_copy) > 0:
        merged_meta['throughput_tiles_per_sec'] = round(total_sr_tiles / (end_copy - start_copy), 3)

    # Write mapping.csv
    with (out_root / 'mapping.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(mapping_header)
        writer.writerows(mapping_rows_out)

    # Write metadata.json
    with (out_root / 'metadata.json').open('w') as f:
        json.dump(merged_meta, f, indent=2)

    print('\n==== MERGE SUMMARY ====')
    print(f'Output dataset: {out_root}')
    print(f'Bases A: {len(bases_a)} | Bases B: {len(bases_b)} (post-dup strategy)')
    if dup:
        print(f'Duplicates encountered: {dup}')
    print(f'Merge strategy: {args.on_duplicate}')
    print(f'Total bases merged: {len(tasks)}')
    print(f'Total SR tiles: {merged_meta["total_sr_tiles"]}')
    if multi_scale:
        for f in factors:
            print(f'  LR x{f} tiles: {merged_meta["total_lr_tiles_per_factor"]["x"+str(f)]}')
        print(f'Total LR tiles (all factors): {merged_meta["total_lr_tiles"]}')
    else:
        print(f'Total LR tiles: {merged_meta["total_lr_tiles"]}')
    print(f"Copy throughput (SR tiles/sec): {merged_meta.get('throughput_tiles_per_sec')}")
    print(f"Duration: {merged_meta['time']['duration_seconds']}s (copy {merged_meta['time']['copy_duration_seconds']}s)")
    print('mapping.csv and metadata.json written.')


if __name__ == '__main__':
    main()
