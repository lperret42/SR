# Super-Resolution Dataset Specification

This document describes the structure and contents of the generated super-resolution (SR) dataset produced by `extract_tile_and_lr_from_jp2.py`.

## Directory Layout
```
<output_root>/
  sr/                      # High-resolution (SR) tiles (original resolution)
    <image_base>/
      <basename>_tile_XXXX_x<X>_y<Y>.png
  lr_x<factor>/             # One directory per low-resolution scale factor (>=2)
    <image_base>/
      <basename>_tile_XXXX_x<X>_y<Y>.png
  mapping.csv               # CSV mapping SR ↔ LR(s) (format depends on #factors)
  metadata.json             # Dataset metadata & provenance
```
Single-scale (legacy) example (factor = 4):
```
output/
  sr/sceneA/sceneA_tile_0003_x1234_y5678.png
  lr_x4/sceneA/sceneA_tile_0003_x1234_y5678.png
  mapping.csv               # 2 columns (sr_path, lr_path)
  metadata.json
```
Multi-scale example (factors = 2 and 4):
```
output/
  sr/sceneA/sceneA_tile_0003_x1234_y5678.png
  lr_x2/sceneA/sceneA_tile_0003_x1234_y5678.png
  lr_x4/sceneA/sceneA_tile_0003_x1234_y5678.png
  mapping.csv               # header: sr_path,lr_x2,lr_x4
  metadata.json
```

## File Naming Convention
`<basename>_tile_<index>_x<X>_y<Y>.png`

- `basename`: Original JP2 filename (without extension)
- `index`: Zero-padded tile index within the extracted grid
- `x`, `y`: Upper-left pixel coordinates of the tile in the original high-res image

## mapping.csv
Two possible formats:

1. Single LR factor (backward compatible):
```
sr_path,lr_path
sr/sceneA/sceneA_tile_0000_x...png,lr_x4/sceneA/sceneA_tile_0000_x...png
```
2. Multi-scale (N factors):
```
sr_path,lr_x2,lr_x4,... (one column per factor, ascending order)
sr/sceneA/sceneA_tile_0000_x...png,lr_x2/sceneA/...,lr_x4/sceneA/...
```
All paths are relative to the dataset root.

## metadata.json
Top-level keys (may extend):
- `requested_images`: Number of images requested
- `selected_images`: Actual number of sampled images (size-filtered)
- `n_tiles_per_image`: Perfect-square tile count per image
- `tile_size`: Side length (pixels) of SR tiles
- `downsample_factor`: (legacy) Single factor when only one is used
- `downsample_factors`: List of factors (present when multi-scale; also present for single for newer versions)
- `degradations`: { `blur_kernel`, `noise_std` }
- `images`: List of per-image objects:
  - `filename`, `width`, `height`, `n_tiles`, `tiles` (list)
  - Each tile (single-scale): { `index`, `hr_path`, `lr_path`, `x`, `y` }
  - Each tile (multi-scale): { `index`, `hr_path`, `lr_paths`, `x`, `y` } where `lr_paths` is a dict: `{ "x2": <path>, "x4": <path>, ... }`
- `total_sr_tiles`, `total_lr_tiles`: (total LR tiles across all factors)
- `total_lr_tiles_per_factor`: Dict factor-key (`"x2"`, `"x4"`, ...) → count
- `skipped_corrupted`, `skipped_geometry_error`
- `output_dirs`: Includes `sr` plus each `lr_x<factor>` directory
- `script_options`: Execution parameters (including `downsample_factors`, seed, workers, min_size_bytes)
- `time`: { `start`, `end`, `duration_seconds` }
- `throughput_tiles_per_sec`: SR tiles per second (not multiplied by factors)

### Example Snippets
Single-scale:
```json
{
  "downsample_factor": 4,
  "downsample_factors": [4],
  "total_sr_tiles": 5000,
  "total_lr_tiles": 5000,
  "total_lr_tiles_per_factor": {"x4": 5000}
}
```
Multi-scale (2 & 4):
```json
{
  "downsample_factors": [2, 4],
  "total_sr_tiles": 5000,
  "total_lr_tiles": 10000,
  "total_lr_tiles_per_factor": {"x2": 5000, "x4": 5000},
  "images": [
    {
      "filename": "sceneA.jp2",
      "tiles": [
        {
          "index": 0,
          "hr_path": "sr/sceneA/...png",
          "lr_paths": {"x2": "lr_x2/sceneA/...png", "x4": "lr_x4/sceneA/...png"}
        }
      ]
    }
  ]
}
```

## Assumptions
- All SR tiles share uniform size = `tile_size`.
- Each LR factor directory contains tiles perfectly aligned (same naming) with SR.
- LR tile spatial size = `tile_size / factor` (integer division guaranteed by generation step).
- Multi-scale: all requested factors are generated for each successful image; if not, that image's tiles are skipped (strict mode in generator).

## Recommended Usage (Training Pipelines)
1. Read `metadata.json` to get `downsample_factors` (fallback to `[downsample_factor]` if older single-scale dataset).
2. Parse `mapping.csv`; if header has more than 2 columns, treat columns 2..N as parallel LR scales.
3. When training multi-scale SR models, choose a factor per batch or create separate dataloaders keyed by factor.
4. Validate expected LR size: `SR.shape == (tile_size, tile_size)` and `LR_f.shape == (tile_size / f, tile_size / f)` for each factor `f`.

## Integrity Checks
- Single-scale: `total_lr_tiles == total_sr_tiles`.
- Multi-scale: `total_lr_tiles == total_sr_tiles * len(downsample_factors)` and per-factor counts equal `total_sr_tiles`.
- `len(mapping.csv lines) - 1 == total_sr_tiles` (SR entries). LR columns must be non-empty.

## Extensibility Ideas
- Add SHA256 checksums per tile.
- Add train/val/test split file lists.
- Add quality metrics (e.g. per-tile variance) for curriculum sampling.
- Add optional per-factor degradations metadata if different degradations are applied per scale in future.

## License & Provenance
Provenance (source filenames) is preserved via `basename`. Any external licensing considerations must be handled upstream of this dataset generation.

---
This specification supports both legacy single-scale and new multi-scale super-resolution datasets.
