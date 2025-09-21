# Super-Resolution Dataset Specification

This document describes the structure and contents of the generated super-resolution (SR) dataset produced by `extract_tile_and_lr_from_jp2.py`.

## Directory Layout
```
<output_root>/
  sr/                     # High-resolution (SR) tiles (original resolution)
    <image_base>/         # One subdirectory per source JP2 (basename without extension)
      <basename>_tile_XXXX_x<X>_y<Y>.png
  lr_x<factor>/           # Low-resolution (LR) tiles (downsampled by <factor>)
    <image_base>/
      <basename>_tile_XXXX_x<X>_y<Y>.png
  mapping.csv             # CSV mapping SR ↔ LR relative paths
  metadata.json           # Dataset metadata & provenance
```

Example (factor = 4):
```
output/
  sr/sceneA/sceneA_tile_0003_x1234_y5678.png
  lr_x4/sceneA/sceneA_tile_0003_x1234_y5678.png
  mapping.csv
  metadata.json
```

## File Naming Convention
`<basename>_tile_<index>_x<X>_y<Y>.png`

- `basename`: Original JP2 filename (without extension)
- `index`: Zero-padded tile index within the extracted grid
- `x`, `y`: Upper-left pixel coordinates of the tile in the original high-res image

## mapping.csv
Two-column CSV (header included):
```
sr_path,lr_path
sr/sceneA/sceneA_tile_0000_x...png,lr_x4/sceneA/sceneA_tile_0000_x...png
...
```
All paths are relative to the dataset root.

## metadata.json
Top-level keys (may extend):
- `requested_images`: Number of images requested
- `selected_images`: Actual number of sampled images (size-filtered)
- `n_tiles_per_image`: Perfect-square tile count per image
- `tile_size`: Side length (pixels) of SR tiles
- `downsample_factor`: Integer LR downsampling factor
- `degradations`: { `blur_kernel`, `noise_std` }
- `images`: List of per-image objects:
  - `filename`, `width`, `height`, `n_tiles`, `tiles` (list)
  - Each tile: { `index`, `hr_path`, `lr_path`, `x`, `y` }
- `total_sr_tiles`, `total_lr_tiles`
- `skipped_corrupted`, `skipped_geometry_error`
- `output_dirs`: Relative SR & LR directory names
- `script_options`: Execution parameters (including seed, workers, min_size_bytes)
- `time`: { `start`, `end`, `duration_seconds` }
- `throughput_tiles_per_sec`: Aggregate processing speed

### Example Snippet
```json
{
  "requested_images": 50,
  "selected_images": 50,
  "n_tiles_per_image": 100,
  "tile_size": 256,
  "downsample_factor": 4,
  "degradations": { "blur_kernel": 5, "noise_std": 2.0 },
  "total_sr_tiles": 5000,
  "time": { "duration_seconds": 412.387 }
}
```

## Assumptions
- All SR tiles share uniform size = `tile_size`.
- LR tiles are produced via integer-factor downsampling + optional Gaussian blur/noise.
- Tile grid per source image is centered and non-overlapping.

## Recommended Usage (Training Pipelines)
1. Parse `metadata.json` for dataset-wide parameters (e.g., `downsample_factor`, `tile_size`).
2. Stream pairs by reading `mapping.csv` and joining with root path.
3. For on-the-fly augmentations, operate on SR tiles only; recompute LR if different degradations are needed.
4. Validate shape consistency: SR tile = `(tile_size, tile_size)`, LR tile = `(tile_size / downsample_factor, tile_size / downsample_factor)`.

## Integrity Checks
- Ensure counts: `total_sr_tiles == len(mapping.csv lines - 1)`.
- Cross-check random spot pairs for coordinate consistency (`x`, `y` fields).

## Extensibility Ideas
- Add SHA256 checksums per tile.
- Add train/val/test split file lists.
- Add quality metrics (e.g. per-tile variance) for curriculum sampling.

## License & Provenance
Provenance (source filenames) is preserved via `basename`. Any external licensing considerations must be handled upstream of this dataset generation.

---
This specification is minimal yet sufficient for integrating into a super-resolution training pipeline.
