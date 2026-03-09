#!/usr/bin/env python3
"""Download pre-trained PhageTransformer model weights and calibration files.

Usage:
    phagetransformer init
    phagetransformer init --model_dir ./my_models/HierDNA
    phagetransformer init --force
"""

import argparse
import hashlib
import logging
import os
import sys
import urllib.request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — update these when releasing new model versions
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = os.path.join(
    os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share')),
    'phagetransformer', 'default',
)

MODEL_FILES = [
    {
        'filename': 'checkpoints/best_aggregator.pt',
        'url': 'https://github.com/yourname/phagetransformer/releases/download/v0.1.0/best_aggregator.pt',
        'sha256': None,    # fill after upload
        'size_mb': 100,
    },
    {
        'filename': 'calibration.json',
        'url': 'https://github.com/yourname/phagetransformer/releases/download/v0.1.0/calibration.json',
        'sha256': None,
        'size_mb': 0.01,
    },
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str, expected_sha256: str = None,
              desc: str = ''):
    """Download a file with progress reporting."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if desc:
        logger.info(f"  Downloading {desc} …")
    else:
        logger.info(f"  Downloading {os.path.basename(dest)} …")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PhageTransformer'})
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            tmp = dest + '.part'
            with open(tmp, 'wb') as f:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1 << 20)
                        total_mb = total / (1 << 20)
                        print(f'\r    {mb:.1f} / {total_mb:.1f} MB ({pct}%)',
                              end='', flush=True)
            if total > 0:
                print()  # newline after progress

    except Exception as e:
        logger.error(f"  Download failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False

    # Verify checksum
    if expected_sha256:
        actual = _sha256(tmp)
        if actual != expected_sha256:
            logger.error(f"  Checksum mismatch for {os.path.basename(dest)}:")
            logger.error(f"    expected: {expected_sha256}")
            logger.error(f"    got:      {actual}")
            os.remove(tmp)
            return False

    os.rename(tmp, dest)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Download pre-trained PhageTransformer model weights',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_dir', type=str, default=None,
                        help=f'Directory to store model files '
                             f'(default: {DEFAULT_MODEL_DIR})')
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if files already exist')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    model_dir = os.path.abspath(model_dir)

    logger.info(f"Model directory: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    total = len(MODEL_FILES)
    total_mb = sum(f['size_mb'] for f in MODEL_FILES)
    logger.info(f"Downloading {total} files (~{total_mb:.0f} MB) …")

    success = 0
    skipped = 0
    for entry in MODEL_FILES:
        dest = os.path.join(model_dir, entry['filename'])

        if os.path.exists(dest) and not args.force:
            # Verify checksum if available
            if entry['sha256'] and _sha256(dest) != entry['sha256']:
                logger.warning(f"  {entry['filename']}: checksum mismatch, "
                               f"re-downloading")
            else:
                logger.info(f"  {entry['filename']}: already exists, skipping")
                skipped += 1
                continue

        ok = _download(
            url=entry['url'],
            dest=dest,
            expected_sha256=entry['sha256'],
            desc=entry['filename'],
        )
        if ok:
            success += 1
        else:
            logger.error(f"  Failed to download {entry['filename']}")

    failed = total - success - skipped
    if failed > 0:
        logger.error(f"\n{failed} file(s) failed to download.")
        logger.error(f"Check your internet connection and try again.")
        sys.exit(1)

    logger.info(f"\nDone. {success} downloaded, {skipped} already present.")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"\nTo run predictions:")
    logger.info(f"  phagetransformer predict --input phages.fasta "
                f"--model_dir {model_dir}")


if __name__ == '__main__':
    main()
