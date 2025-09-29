#!/usr/bin/env python3
"""Download PCRS tiles individually or via a batch JSON specification.

Features
--------
- Download a rectangular block of tiles defined by a top-left reference tile, width, and height.
- Batch mode: pass a JSON file mapping city names to [top_left, bottom_right] tile IDs.
- Retries with exponential backoff on network/server errors.
- Organises outputs per city (batch) and writes a metadata manifest describing all downloaded JP2 files.
"""

import argparse
import os
import requests
from pathlib import Path
import time
from tqdm import tqdm
import zipfile
import tempfile
import json
from datetime import datetime


def parse_tile_name(tile_name):
    """Parse a tile name and return its components."""
    parts = tile_name.split('-')
    if len(parts) < 4:
        raise ValueError(f"Invalid tile name: {tile_name}")
    year = parts[0]
    x_coord = int(parts[1])
    y_coord = int(parts[2])
    metadata = '-'.join(parts[3:])
    return year, x_coord, y_coord, metadata


def generate_tile_name(year, x_coord, y_coord, metadata):
    """Create a tile name from components."""
    return f"{year}-{x_coord:05d}-{y_coord:05d}-{metadata}"


def generate_tile_grid(reference_tile, width_x, height_y):
    """Generate the tile names for a rectangular grid."""
    year, ref_x, ref_y, metadata = parse_tile_name(reference_tile)
    tiles = []
    for row in range(height_y):
        for col in range(width_x):
            x_coord = ref_x + (col * 2)
            y_coord = ref_y - (row * 2)
            tile_name = generate_tile_name(year, x_coord, y_coord, metadata)
            tiles.append(tile_name)
    return tiles


def construct_download_url(tile_name):
    """Return the download endpoint."""
    return "https://pcrs.ign.fr/download"


def get_dalle_payload(tile_name):
    """Build the POST payload for a tile."""
    year, x_coord, y_coord, metadata = parse_tile_name(tile_name)
    x_min = x_coord * 100
    y_max = y_coord * 100
    x_max = x_min + 200
    y_min = y_max - 200
    dalle_name = f"{x_min}-{y_min}-{x_max}-{y_max}-{year}-{metadata}"
    return {"dalle[]": dalle_name}


def download_tile(tile_name, output_dir, session, retries=5, retry_wait=5.0, retry_backoff=1.5):
    """Download a tile with retry logic.

    Returns:
        (success, file_size_bytes, status_msg, output_paths)
    """
    url = construct_download_url(tile_name)
    payload = get_dalle_payload(tile_name)

    existing_files = list(output_dir.glob(f"{tile_name}.*"))
    if existing_files:
        total_size = sum(f.stat().st_size for f in existing_files)
        return True, total_size, "Déjà téléchargé", existing_files

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            print(f"🔗 URL: {url} (tentative {attempt}/{retries})")
            print(f"📦 Payload: {payload}")
            response = session.post(url, data=payload, stream=True, timeout=60)
            status_code = response.status_code
            if status_code >= 500 or status_code == 429:
                raise requests.HTTPError(f"Statut {status_code}")
            if status_code >= 400:
                raise requests.HTTPError(f"Statut {status_code}")

            content_type = response.headers.get('content-type', '')
            print(f"📄 Content-Type: {content_type}")

            file_size = 0
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
                    file_size += len(chunk)

            is_zip = content.startswith(b'PK\x03\x04') or content_type == 'application/zip'
            if is_zip:
                print("📦 ZIP détecté - extraction en cours...")
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                    temp_zip.write(content)
                    temp_zip_path = temp_zip.name
                output_paths = []
                try:
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        file_list = zip_ref.namelist()
                        print(f"📋 Contenu du ZIP: {file_list}")
                        extracted_size = 0
                        for file_info in zip_ref.infolist():
                            file_ext = Path(file_info.filename).suffix
                            output_filename = f"{tile_name}{file_ext}"
                            output_path = output_dir / output_filename
                            with zip_ref.open(file_info) as source, open(output_path, 'wb') as target:
                                target.write(source.read())
                            extracted_size += output_path.stat().st_size
                            output_paths.append(output_path)
                            print(f"✅ Extrait: {output_filename}")
                        return True, extracted_size, "Téléchargé et extrait", output_paths
                finally:
                    os.unlink(temp_zip_path)
            else:
                is_jp2 = content.startswith(b'\x00\x00\x00\x0cjP  ') or content.startswith(b'\xff\x4f\xff\x51')
                is_tif = content.startswith(b'II*\x00') or content.startswith(b'MM\x00*')
                if is_jp2:
                    final_path = output_dir / f"{tile_name}.jp2"
                    format_detected = "JPEG 2000"
                elif is_tif:
                    final_path = output_dir / f"{tile_name}.tif"
                    format_detected = "TIFF"
                else:
                    final_path = output_dir / f"{tile_name}.bin"
                    format_detected = "Format inconnu"
                    print(f"⚠️  Format non reconnu - début: {content[:20].hex()}")
                print(f"🔍 Format détecté: {format_detected}")
                print(f"💾 Sauvegarde: {final_path.name}")
                with open(final_path, 'wb') as f:
                    f.write(content)
                return True, file_size, "Téléchargé", [final_path]
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, zipfile.BadZipFile) as e:
            if attempt >= retries:
                return False, 0, f"Échec après {retries} tentatives: {e}", []
            wait_time = retry_wait * (retry_backoff ** (attempt - 1))
            print(f"⚠️  Erreur tentative {attempt}/{retries} pour {tile_name}: {e} -> attente {wait_time:.1f}s avant retry")
            time.sleep(wait_time)
        except Exception as e:
            return False, 0, f"Erreur non gérée: {e}", []
    return False, 0, "Échec inconnu", []


def format_bytes(bytes_size):
    """Human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def compute_dimensions_from_corners(top_left, bottom_right):
    """Compute width/height (in tiles) between two corner tiles."""
    tl_year, tl_x, tl_y, tl_meta = parse_tile_name(top_left)
    br_year, br_x, br_y, br_meta = parse_tile_name(bottom_right)
    if tl_year != br_year or tl_meta != br_meta:
        raise ValueError("Top-left and bottom-right tiles must share year and metadata")
    if br_x < tl_x or br_y > tl_y:
        raise ValueError("Bottom-right tile must be to the bottom-right of the top-left tile")
    if (br_x - tl_x) % 2 != 0 or (tl_y - br_y) % 2 != 0:
        raise ValueError("Tile coordinates are not aligned on 2-unit grid")
    width = (br_x - tl_x) // 2 + 1
    height = (tl_y - br_y) // 2 + 1
    return width, height


def sanitize_city_name(name):
    return name.replace(' ', '_')


def process_grid(reference_tile, width_x, height_y, output_dir, session, args, root_output, label=None):
    """Download a rectangular tile grid and return statistics & metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles = generate_tile_grid(reference_tile, width_x, height_y)
    total_tiles = len(tiles)

    year, ref_x, ref_y, metadata = parse_tile_name(reference_tile)
    max_x = ref_x + ((width_x - 1) * 2)
    min_y = ref_y - ((height_y - 1) * 2)

    print("=" * 60)
    header = label if label else "TÉLÉCHARGEUR DE TUILES PCRS"
    print(f"🗺️  {header}")
    print("=" * 60)
    print(f"📍 Tuile de référence: {reference_tile}")
    print(f"📏 Dimensions: {width_x} × {height_y} tuiles")
    print(f"📁 Répertoire de sortie: {output_dir.absolute()}")
    print()
    print("🔍 Génération de la grille de tuiles...")
    print(f"📊 Nombre total de tuiles à télécharger: {total_tiles}")
    print("🗺️  Zone couverte:")
    print(f"   X: {ref_x} → {max_x} (Lambert 93)")
    print(f"   Y: {min_y} → {ref_y} (Lambert 93)")
    print()

    start_time = time.time()
    total_downloaded = 0
    total_skipped = 0
    total_errors = 0
    total_size = 0
    tile_records = []

    with tqdm(tiles, desc="Téléchargement", unit="tuile") as pbar:
        for tile_name in pbar:
            pbar.set_postfix_str(f"Actuel: {tile_name[:20]}...")
            success, file_size, status, paths = download_tile(
                tile_name,
                output_dir,
                session,
                retries=args.retries,
                retry_wait=args.retry_wait,
                retry_backoff=args.retry_backoff,
            )
            if success:
                total_size += file_size
                if status.startswith("Téléchargé"):
                    total_downloaded += 1
                elif status == "Déjà téléchargé":
                    total_skipped += 1
            else:
                total_errors += 1
                print(f"❌ Erreur pour {tile_name}: {status}")
            pbar.set_postfix_str(
                f"✅ {total_downloaded} | ⏭️  {total_skipped} | ❌ {total_errors}"
            )
            rel_paths = [str(p.resolve().relative_to(root_output)) for p in paths]
            tile_records.append({
                "tile_name": tile_name,
                "status": status,
                "success": success,
                "size_bytes": file_size,
                "files": rel_paths,
            })

    end_time = time.time()
    duration = end_time - start_time

    print()
    print("=" * 60)
    print("📈 RÉSUMÉ DU TÉLÉCHARGEMENT")
    print("=" * 60)
    print(f"⏱️  Durée totale: {duration:.1f} secondes")
    print(f"✅ Tuiles téléchargées: {total_downloaded}")
    print(f"⏭️  Tuiles déjà présentes: {total_skipped}")
    print(f"❌ Erreurs: {total_errors}")
    print(f"📊 Total traité: {total_downloaded + total_skipped + total_errors}/{total_tiles}")
    print(f"💾 Taille totale: {format_bytes(total_size)}")
    if total_downloaded > 0:
        avg_speed = total_downloaded / duration if duration > 0 else float('inf')
        print(f"⚡ Vitesse moyenne: {avg_speed:.1f} tuiles/seconde")
        avg_size = total_size / (total_downloaded + total_skipped) if (total_downloaded + total_skipped) else 0
        print(f"📏 Taille moyenne par tuile: {format_bytes(avg_size)}")
    print()
    if total_errors == 0:
        print("🎉 Téléchargement terminé avec succès !")
    else:
        print(f"⚠️  Téléchargement terminé avec {total_errors} erreurs")
    print(f"📁 Fichiers disponibles dans: {output_dir.absolute()}\n")

    return {
        "reference_tile": reference_tile,
        "width_tiles": width_x,
        "height_tiles": height_y,
        "total_tiles": total_tiles,
        "downloaded": total_downloaded,
        "skipped": total_skipped,
        "errors": total_errors,
        "size_bytes": total_size,
        "duration_seconds": round(duration, 3),
        "tiles": tile_records,
        "bbox": {
            "x_min": ref_x,
            "x_max": max_x,
            "y_min": min_y,
            "y_max": ref_y,
            "year": year,
            "metadata": metadata,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Télécharge un rectangle de tuiles PCRS (mode simple ou batch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python %(prog)s 2020-04960-66124-LA93-0M05-RVB 5 3 ./tiles/
  python %(prog)s --batch-json pcrs.json ./root_output/
        """
    )

    parser.add_argument('reference_tile', nargs='?', help='Tuile de référence (coin supérieur gauche du rectangle)')
    parser.add_argument('width_x', nargs='?', type=int, help='Largeur (nombre de tuiles sur axe X)')
    parser.add_argument('height_y', nargs='?', type=int, help='Hauteur (nombre de tuiles sur axe Y)')
    parser.add_argument('output_dir', nargs='?', type=Path, help='Répertoire de sortie (ou racine batch)')

    parser.add_argument('--batch-json', type=Path, default=None,
                        help='Fichier JSON mapping ville -> [tile_haut_gauche, tile_bas_droite]')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='(Compatibilité) Argument conservé mais non utilisé directement')
    parser.add_argument('--retries', type=int, default=5, help='Nombre de tentatives par tuile (défaut: 5)')
    parser.add_argument('--retry-wait', type=float, default=5.0,
                        help="Attente initiale avant retry (sec, défaut: 5.0)")
    parser.add_argument('--retry-backoff', type=float, default=1.5,
                        help="Facteur de backoff exponentiel (défaut: 1.5)")
    parser.add_argument('--dry-run', action='store_true', help='Ne pas télécharger, afficher seulement la sélection (mode simple uniquement)')

    args = parser.parse_args()

    if args.batch_json:
        if args.output_dir is None:
            parser.error('output_dir est requis en mode batch (--batch-json)')
    else:
        missing = [name for name, value in [('reference_tile', args.reference_tile),
                                            ('width_x', args.width_x),
                                            ('height_y', args.height_y),
                                            ('output_dir', args.output_dir)] if value is None]
        if missing:
            parser.error(f"Arguments manquants: {', '.join(missing)}")
    return args


def load_batch_config(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('Le JSON batch doit être un objet dict ville -> [top_left, bottom_right]')
    normalized = {}
    for city, tiles in data.items():
        if not isinstance(tiles, list) or len(tiles) != 2:
            raise ValueError(f"Entrée invalide pour {city}: attendu [top_left, bottom_right]")
        normalized[city] = tiles
    return normalized


def write_metadata(root_output, manifest):
    output_path = root_output / 'download_metadata.json'
    with output_path.open('w') as f:
        json.dump(manifest, f, indent=2)
    print(f"📄 Metadata écrite dans {output_path}")


def main():
    args = parse_args()

    session = requests.Session()
    session.headers.update({'User-Agent': 'PCRS-Downloader/2.0'})

    if args.batch_json:
        root_output = args.output_dir.resolve()
        root_output.mkdir(parents=True, exist_ok=True)
        batch_config = load_batch_config(args.batch_json)
        manifest = {
            'mode': 'batch',
            'generated_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'root_output': str(root_output),
            'sources': str(args.batch_json),
            'cities': [],
            'totals': {
                'tiles_requested': 0,
                'downloaded': 0,
                'skipped': 0,
                'errors': 0,
                'size_bytes': 0,
            }
        }

        for city, (top_left, bottom_right) in batch_config.items():
            print(f"\n=== Traitement de la ville: {city} ===")
            width, height = compute_dimensions_from_corners(top_left, bottom_right)
            city_dir = root_output / sanitize_city_name(city)
            summary = process_grid(
                reference_tile=top_left,
                width_x=width,
                height_y=height,
                output_dir=city_dir,
                session=session,
                args=args,
                root_output=root_output,
                label=f"Téléchargement PCRS - {city}",
            )
            summary.update({
                'city': city,
                'top_left': top_left,
                'bottom_right': bottom_right,
            })
            manifest['cities'].append(summary)
            manifest['totals']['tiles_requested'] += summary['total_tiles']
            manifest['totals']['downloaded'] += summary['downloaded']
            manifest['totals']['skipped'] += summary['skipped']
            manifest['totals']['errors'] += summary['errors']
            manifest['totals']['size_bytes'] += summary['size_bytes']

        write_metadata(root_output, manifest)
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            width = args.width_x
            height = args.height_y
            tiles = generate_tile_grid(args.reference_tile, width, height)
            print(f"Dry run: {len(tiles)} tuiles seraient téléchargées vers {output_dir}")
            return

        summary = process_grid(
            reference_tile=args.reference_tile,
            width_x=args.width_x,
            height_y=args.height_y,
            output_dir=output_dir,
            session=session,
            args=args,
            root_output=output_dir,
            label="TÉLÉCHARGEUR DE TUILES PCRS",
        )

        manifest = {
            'mode': 'single',
            'generated_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'root_output': str(output_dir),
            'summary': summary,
        }
        write_metadata(output_dir, manifest)


if __name__ == "__main__":
    main()
