#!/usr/bin/env python3
"""Download host reference genomes for the genera present in a train.csv.

Pipeline
--------
1. Read ``train.csv`` and collect the unique host genera. The genus is the
   last ``;``-separated element of each (``|``-delimited, multi-label)
   ``host_labels`` cell — i.e. the same convention ``genus_to_idx`` uses in
   train.py (``label.split(';')[-1]``).

2. Download the GTDB metadata file (``bac120_metadata_r226.tsv.gz`` by
   default) from the configured mirror, unless a cached copy already exists.

3. Stream-parse the gzipped metadata, keeping only the columns we need
   (accession, gtdb_taxonomy, gtdb_representative) and only the rows that are
   species representatives (``gtdb_representative == 't'``) whose genus is one
   of the targets. This yields *all* species representatives for every genus
   represented in the training data.

4. For each selected accession, download its genomic FASTA from the NCBI
   Datasets v2 REST API, and write it gzipped to ``<output_dir>/genomes/``.
   Already-present genomes are skipped, so the script is resumable. Reps are
   randomly subsampled per genus to ``--max_genomes_per_genus`` (default 20).

5. Write ``host_genome_manifest.tsv`` (the file BacterialGenomeStore reads),
   with ``species`` and ``genome_path`` columns plus ``genus``/``accession``
   for traceability. The ``species`` field's first token is the GTDB genus,
   because the store derives a genome's genus via ``species.split()[0]``.

Genera with no GTDB representative are written to ``missing_genera.txt`` and
skipped (these are the labels that will have no bacterial genome data).

Note: the sandbox this was written in blocks NCBI/GTDB, so the download paths
have not been exercised live — run on a machine with network access.
"""

import argparse
import csv
import gzip
import io
import os
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile

DEFAULT_GTDB_BASE = ("https://data.gtdb.aau.ecogenomic.org/releases/"
                     "release226/226.0/")
DEFAULT_METADATA = "bac120_metadata_r226.tsv.gz"
NCBI_DOWNLOAD_URL = ("https://api.ncbi.nlm.nih.gov/datasets/v2/genome/"
                     "accession/{acc}/download"
                     "?include_annotation_type=GENOME_FASTA")
USER_AGENT = "host-genome-downloader/1.0"


# ---------------------------------------------------------------------------
# train.csv -> target genera
# ---------------------------------------------------------------------------

def collect_target_genera(train_csv, label_col="host_labels",
                          label_delim="|"):
    import pandas as pd
    df = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
    if label_col not in df.columns:
        raise SystemExit(
            f"Column '{label_col}' not found in {train_csv}; "
            f"columns: {list(df.columns)}")
    genera = set()
    for cell in df[label_col]:
        for lab in str(cell).split(label_delim):
            lab = lab.strip()
            if not lab:
                continue
            genus = lab.split(';')[-1].strip()
            if genus:
                genera.add(genus)
    return genera


# ---------------------------------------------------------------------------
# GTDB metadata download + parse
# ---------------------------------------------------------------------------

def _http_get(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def ensure_metadata(base_url, filename, cache_dir, force=False, timeout=120):
    """Download the GTDB metadata file to cache_dir if absent. Returns path."""
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, filename)
    if os.path.exists(dest) and not force:
        print(f"  metadata cached: {dest} "
              f"({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    url = base_url.rstrip('/') + '/' + filename
    tmp = dest + ".part"
    print(f"  downloading metadata: {url}")
    with _http_get(url, timeout) as resp, open(tmp, 'wb') as out:
        total = int(resp.headers.get('Content-Length', 0))
        done = 0
        chunk = 1 << 20
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if total:
                sys.stdout.write(
                    f"\r    {done/1e6:.1f}/{total/1e6:.1f} MB "
                    f"({100*done/total:.0f}%)")
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(tmp, dest)
    return dest


def _strip_rank(tok):
    tok = tok.strip()
    return tok[3:] if len(tok) >= 3 and tok[1:3] == '__' else tok


def iter_representatives(metadata_path, target_genera,
                         acc_col="accession",
                         tax_col="gtdb_taxonomy",
                         rep_col="gtdb_representative"):
    """Yield (accession, genus, species) for species-representative rows whose
    genus is in ``target_genera``. ``accession`` has the GTDB ``RS_``/``GB_``
    prefix stripped (ready for the NCBI API)."""
    with gzip.open(metadata_path, 'rt', newline='') as fh:
        reader = csv.reader(fh, delimiter='\t')
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for col in (acc_col, tax_col, rep_col):
            if col not in idx:
                raise SystemExit(
                    f"Column '{col}' not in metadata header. Available "
                    f"(first 25): {header[:25]} … use --acc_col/--tax_col/"
                    f"--rep_col to override.")
        ai, ti, ri = idx[acc_col], idx[tax_col], idx[rep_col]
        for row in reader:
            if row[ri].strip().lower() not in ('t', 'true', '1'):
                continue
            tax = row[ti]
            genus = species = None
            for tok in tax.split(';'):
                tok = tok.strip()
                if tok.startswith('g__'):
                    genus = _strip_rank(tok)
                elif tok.startswith('s__'):
                    species = _strip_rank(tok)
            if not genus or genus not in target_genera:
                continue
            acc = row[ai].strip()
            if acc.startswith(('RS_', 'GB_')):
                acc = acc[3:]
            if not species:
                species = f"{genus} {acc}"
            yield acc, genus, species


# ---------------------------------------------------------------------------
# NCBI genome download (Datasets v2 REST API)
# ---------------------------------------------------------------------------

def download_genome(accession, out_path, timeout=120, max_retries=4,
                    sleep=0.34, api_key=None):
    """Download one assembly's genomic FASTA and write it gzipped to out_path.

    Returns True on success, False if the accession could not be retrieved.
    """
    url = NCBI_DOWNLOAD_URL.format(acc=accession)
    if api_key:
        url += f"&api_key={api_key}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/zip"}

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            # The download endpoint returns a zip; a JSON body means error.
            if not payload[:2] == b'PK':
                raise ValueError("response was not a zip (likely no such "
                                 "accession or rate-limited)")
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                fna = [n for n in zf.namelist()
                       if n.endswith(('_genomic.fna', '.fna'))
                       and '/data/' in n]
                if not fna:
                    raise ValueError("no genomic FASTA in download")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp = out_path + ".part"
                with zf.open(fna[0]) as src, gzip.open(tmp, 'wb') as dst:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf:
                            break
                        dst.write(buf)
                os.replace(tmp, out_path)
            time.sleep(sleep)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
                zipfile.BadZipFile) as e:
            wait = sleep * (2 ** attempt)
            print(f"    [{accession}] attempt {attempt}/{max_retries} "
                  f"failed: {e}; retrying in {wait:.1f}s")
            time.sleep(wait)
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('train_csv', help='Path to train.csv')
    p.add_argument('-o', '--output_dir', default=None,
                   help='Output dir (default: host_genomes/ next to train.csv)')
    p.add_argument('--gtdb_base_url', default=DEFAULT_GTDB_BASE)
    p.add_argument('--metadata_file', default=DEFAULT_METADATA)
    p.add_argument('--archaea_metadata', default=None,
                   help='Also parse this archaeal metadata file, e.g. '
                        'ar53_metadata_r226.tsv.gz')
    p.add_argument('--cache_dir', default=None,
                   help='Where to cache metadata (default: output_dir)')
    p.add_argument('--max_genomes_per_genus', type=int, default=20,
                   help='Randomly subsample species reps to this many per '
                        'genus (0 = keep all reps)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for per-genus subsampling')
    p.add_argument('--label_col', default='host_labels')
    p.add_argument('--label_delim', default='|')
    p.add_argument('--acc_col', default='accession')
    p.add_argument('--tax_col', default='gtdb_taxonomy')
    p.add_argument('--rep_col', default='gtdb_representative')
    p.add_argument('--ncbi_api_key', default=None)
    p.add_argument('--sleep', type=float, default=0.34,
                   help='Seconds between NCBI requests (lower with --ncbi_api_key)')
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--max_retries', type=int, default=4)
    p.add_argument('--force_metadata', action='store_true')
    p.add_argument('--dry_run', action='store_true',
                   help='Resolve accessions and write the manifest, but skip '
                        'the actual genome downloads')
    args = p.parse_args()

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.train_csv)), 'host_genomes')
    cache_dir = args.cache_dir or out_dir
    genomes_dir = os.path.join(out_dir, 'genomes')
    os.makedirs(genomes_dir, exist_ok=True)

    genera = collect_target_genera(args.train_csv, args.label_col,
                                   args.label_delim)
    print(f"  {len(genera)} unique host genera in {args.train_csv}")

    # Resolve genus -> [(accession, species)] from GTDB metadata.
    meta_files = [ensure_metadata(args.gtdb_base_url, args.metadata_file,
                                  cache_dir, args.force_metadata, args.timeout)]
    if args.archaea_metadata:
        meta_files.append(ensure_metadata(
            args.gtdb_base_url, args.archaea_metadata, cache_dir,
            args.force_metadata, args.timeout))

    by_genus = {}
    for mf in meta_files:
        print(f"  parsing {os.path.basename(mf)} …")
        for acc, genus, species in iter_representatives(
                mf, genera, args.acc_col, args.tax_col, args.rep_col):
            by_genus.setdefault(genus, []).append((acc, species))

    found = set(by_genus)
    missing = sorted(genera - found)
    n_reps = sum(len(v) for v in by_genus.values())
    print(f"  matched {len(found)}/{len(genera)} genera "
          f"({n_reps} species representatives); {len(missing)} missing")
    if missing:
        with open(os.path.join(out_dir, 'missing_genera.txt'), 'w') as f:
            f.write('\n'.join(missing) + '\n')
        print(f"  missing genera -> {os.path.join(out_dir, 'missing_genera.txt')}")

    # Apply the per-genus cap (random subsample) up front so we can report
    # how many genomes will be fetched before starting.
    rng = random.Random(args.seed)
    cap = args.max_genomes_per_genus
    selected = {}  # genus -> [(accession, species), ...]
    for genus in sorted(by_genus):
        reps = by_genus[genus]
        if cap > 0 and len(reps) > cap:
            reps = rng.sample(reps, cap)
        selected[genus] = reps

    n_selected = sum(len(v) for v in selected.values())
    n_present = sum(
        os.path.exists(os.path.join(genomes_dir, f'{acc}.fna.gz'))
        for v in selected.values() for acc, _ in v)
    n_todo = n_selected - n_present
    cap_note = f", \u2264{cap}/genus" if cap > 0 else ""
    print(f"  selected {n_selected} genomes across {len(selected)} genera"
          f"{cap_note}")
    verb = "would download" if args.dry_run else "will download"
    print(f"  {verb} {n_todo} genomes ({n_present} already present, skipped)")

    # Download + manifest.
    manifest_path = os.path.join(out_dir, 'host_genome_manifest.tsv')
    n_ok = n_skip = n_fail = 0
    with open(manifest_path, 'w', newline='') as mf:
        w = csv.writer(mf, delimiter='\t')
        w.writerow(['species', 'genome_path', 'genus', 'accession'])
        for genus in sorted(selected):
            for acc, species in selected[genus]:
                rel = os.path.join('genomes', f'{acc}.fna.gz')
                dest = os.path.join(out_dir, rel)
                if os.path.exists(dest):
                    n_skip += 1
                elif args.dry_run:
                    pass
                else:
                    ok = download_genome(
                        acc, dest, timeout=args.timeout,
                        max_retries=args.max_retries, sleep=args.sleep,
                        api_key=args.ncbi_api_key)
                    if ok:
                        n_ok += 1
                        print(f"    [{genus}] {acc} ok")
                    else:
                        n_fail += 1
                        print(f"    [{genus}] {acc} FAILED — omitted from manifest")
                        continue
                w.writerow([species, rel, genus, acc])

    print(f"\n  downloaded {n_ok}, skipped {n_skip} (already present), "
          f"failed {n_fail}")
    print(f"  manifest -> {manifest_path}")
    print(f"  point training at: --host_genome_dir {out_dir}")


if __name__ == '__main__':
    main()
