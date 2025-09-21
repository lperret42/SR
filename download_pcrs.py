#!/usr/bin/env python3
"""
Script pour télécharger un rectangle de tuiles depuis l'interface PCRS géoservices.
"""

import argparse
import os
import requests
from pathlib import Path
import time
from tqdm import tqdm
import zipfile
import tempfile


def parse_tile_name(tile_name):
    """Parse le nom d'une tuile et retourne ses composants."""
    parts = tile_name.split('-')
    year = parts[0]
    x_coord = int(parts[1])
    y_coord = int(parts[2])
    metadata = '-'.join(parts[3:])  # LA93-0M05-RVB
    
    return year, x_coord, y_coord, metadata


def generate_tile_name(year, x_coord, y_coord, metadata):
    """Génère un nom de tuile à partir de ses composants."""
    return f"{year}-{x_coord:05d}-{y_coord:05d}-{metadata}"


def generate_tile_grid(reference_tile, width_x, height_y):
    """
    Génère la grille de noms de tuiles.
    
    Args:
        reference_tile: Tuile de référence (coin supérieur gauche)
        width_x: Nombre de tuiles en largeur (axe X, gauche vers droite)
        height_y: Nombre de tuiles en hauteur (axe Y, haut vers bas)
    
    Returns:
        Liste des noms de tuiles
    """
    year, ref_x, ref_y, metadata = parse_tile_name(reference_tile)
    
    tiles = []
    for row in range(height_y):
        for col in range(width_x):
            x_coord = ref_x + (col * 2)  # +2 vers la droite
            y_coord = ref_y - (row * 2)  # -2 vers le bas
            tile_name = generate_tile_name(year, x_coord, y_coord, metadata)
            tiles.append(tile_name)
    
    return tiles


def construct_download_url(tile_name):
    """Construit l'URL de téléchargement pour une tuile."""
    return "https://pcrs.ign.fr/download"


def get_dalle_payload(tile_name):
    """Construit la payload dalle[] pour le POST."""
    year, x_coord, y_coord, metadata = parse_tile_name(tile_name)
    
    # Conversion vers les vraies coordonnées Lambert 93
    x_min = x_coord * 100  # 04960 → 496000
    y_max = y_coord * 100  # 66124 → 6612400
    x_max = x_min + 200    # Chaque tuile fait 200m
    y_min = y_max - 200    # Y décroit vers le bas
    
    # Format de la dalle: x_min-y_min-x_max-y_max-year-metadata
    dalle_name = f"{x_min}-{y_min}-{x_max}-{y_max}-{year}-{metadata}"
    
    return {"dalle[]": dalle_name}


def download_tile(tile_name, output_dir, session, retries=5, retry_wait=5.0, retry_backoff=1.5):
    """
    Télécharge une tuile dans le répertoire de sortie avec gestion de retries.

    Args:
        tile_name: nom de la tuile
        output_dir: dossier de sortie
        session: session requests
        retries: nombre total de tentatives
        retry_wait: attente initiale entre tentatives (s)
        retry_backoff: facteur multiplicatif pour backoff exponentiel

    Returns:
        tuple: (success: bool, file_size: int, status: str)
    """
    url = construct_download_url(tile_name)
    payload = get_dalle_payload(tile_name)

    # Vérifier si déjà extrait (chercher les .jp2 / .tif / .jp2.zip déjà extraits)
    existing_files = list(output_dir.glob(f"{tile_name}.*"))
    if existing_files:
        total_size = sum(f.stat().st_size for f in existing_files)
        return True, total_size, "Déjà téléchargé"

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            print(f"🔗 URL: {url} (tentative {attempt}/{retries})")
            print(f"📦 Payload: {payload}")
            response = session.post(url, data=payload, stream=True, timeout=60)
            status_code = response.status_code
            # Retenter sur 5xx ou 429
            if status_code >= 500 or status_code in (429,):
                raise requests.HTTPError(f"Statut {status_code}")
            # Autres erreurs HTTP
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
                print(f"📦 ZIP détecté - extraction en cours...")
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                    temp_zip.write(content)
                    temp_zip_path = temp_zip.name
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
                            print(f"✅ Extrait: {output_filename}")
                        return True, extracted_size, "Téléchargé et extrait"
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
                return True, file_size, "Téléchargé"
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, zipfile.BadZipFile) as e:
            if attempt >= retries:
                return False, 0, f"Échec après {retries} tentatives: {e}"
            wait_time = retry_wait * (retry_backoff ** (attempt - 1))
            print(f"⚠️  Erreur tentative {attempt}/{retries} pour {tile_name}: {e} -> attente {wait_time:.1f}s avant retry")
            time.sleep(wait_time)
        except Exception as e:
            # Erreur non prévue, ne pas boucler indéfiniment
            return False, 0, f"Erreur non gérée: {e}"
    return False, 0, "Échec inconnu"


def format_bytes(bytes_size):
    """Formate une taille en bytes de manière lisible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge un rectangle de tuiles PCRS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python %(prog)s 2020-04960-66124-LA93-0M05-RVB 5 3 ./tiles/
  python %(prog)s 2020-04960-66124-LA93-0M05-RVB 10 10 /data/pcrs_tiles/
        """
    )
    
    parser.add_argument(
        'reference_tile',
        help='Tuile de référence (coin supérieur gauche du rectangle)'
    )
    parser.add_argument(
        'width_x',
        type=int,
        help='Largeur du rectangle (nombre de tuiles sur axe X, gauche vers droite)'
    )
    parser.add_argument(
        'height_y',
        type=int,
        help='Hauteur du rectangle (nombre de tuiles sur axe Y, haut vers bas)'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Répertoire de sortie pour les tuiles téléchargées'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Nombre maximum de téléchargements simultanés (défaut: 4)'
    )
    parser.add_argument(
        '--retries', type=int, default=5,
        help='Nombre de tentatives par tuile avant abandon (défaut: 5)'
    )
    parser.add_argument(
        '--retry-wait', type=float, default=5.0,
        help="Attente initiale (sec) avant la première relance (défaut: 5.0)"
    )
    parser.add_argument(
        '--retry-backoff', type=float, default=1.5,
        help="Facteur multiplicatif de backoff exponentiel (défaut: 1.5)"
    )
    
    args = parser.parse_args()
    
    # Validation des arguments
    assert args.width_x > 0, "La largeur doit être positive"
    assert args.height_y > 0, "La hauteur doit être positive"
    
    # Création du répertoire de sortie
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🗺️  TÉLÉCHARGEUR DE TUILES PCRS")
    print("=" * 60)
    print(f"📍 Tuile de référence: {args.reference_tile}")
    print(f"📏 Dimensions: {args.width_x} × {args.height_y} tuiles")
    print(f"📁 Répertoire de sortie: {args.output_dir.absolute()}")
    print()
    
    # Génération de la grille de tuiles
    print("🔍 Génération de la grille de tuiles...")
    tiles = generate_tile_grid(args.reference_tile, args.width_x, args.height_y)
    total_tiles = len(tiles)
    
    print(f"📊 Nombre total de tuiles à télécharger: {total_tiles}")
    
    # Affichage des coordonnées extrêmes
    year, ref_x, ref_y, metadata = parse_tile_name(args.reference_tile)
    max_x = ref_x + ((args.width_x - 1) * 2)
    min_y = ref_y - ((args.height_y - 1) * 2)
    
    print(f"🗺️  Zone couverte:")
    print(f"   X: {ref_x} → {max_x} (Lambert 93)")
    print(f"   Y: {min_y} → {ref_y} (Lambert 93)")
    print()
    
    # Initialisation des statistiques
    start_time = time.time()
    total_downloaded = 0
    total_skipped = 0
    total_errors = 0
    total_size = 0
    
    # Session pour réutiliser les connexions
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'PCRS-Downloader/1.0'
    })
    
    print("🚀 Début des téléchargements...")
    print()
    
    # Barre de progression
    with tqdm(tiles, desc="Téléchargement", unit="tuile") as pbar:
        for tile_name in pbar:
            pbar.set_postfix_str(f"Actuel: {tile_name[:20]}...")
            success, file_size, status = download_tile(
                tile_name, args.output_dir, session,
                retries=args.retries,
                retry_wait=args.retry_wait,
                retry_backoff=args.retry_backoff
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
    
    # Statistiques finales
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
        avg_speed = total_downloaded / duration
        print(f"⚡ Vitesse moyenne: {avg_speed:.1f} tuiles/seconde")
        avg_size = total_size / (total_downloaded + total_skipped)
        print(f"📏 Taille moyenne par tuile: {format_bytes(avg_size)}")
    
    print()
    if total_errors == 0:
        print("🎉 Téléchargement terminé avec succès !")
    else:
        print(f"⚠️  Téléchargement terminé avec {total_errors} erreurs")
    
    print(f"📁 Fichiers disponibles dans: {args.output_dir.absolute()}")


if __name__ == "__main__":
    main()
