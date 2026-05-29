#!/usr/bin/env python3
"""Extract genomes used in the importance analysis from the training FASTA.

Usage:
    python extract_importance_genomes.py \
        --tsv importance_results/importance.tsv \
        --fasta /path/to/train.fna.gz \
        --output importance_genomes.fna.gz
"""

import argparse
import csv
import gzip
from Bio import SeqIO


def main():
    parser = argparse.ArgumentParser(
        description='Extract genomes used in importance analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', required=True,
                        help='importance.tsv from the scrambling analysis')
    parser.add_argument('--fasta', required=True,
                        help='Training FASTA (plain or gzipped)')
    parser.add_argument('--output', '-o', default='importance_genomes.fna.gz',
                        help='Output FASTA (gzipped if .gz)')

    args = parser.parse_args()

    # Collect unique genome IDs from TSV
    ids = set()
    with open(args.tsv) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            ids.add(row['genome_id'])
    print(f"Found {len(ids)} unique genome IDs in {args.tsv}")

    # Filter FASTA
    opener = gzip.open if args.fasta.endswith('.gz') else open
    out_opener = gzip.open if args.output.endswith('.gz') else open

    written = 0
    sanitized = 0
    with opener(args.fasta, 'rt') as fh_in, out_opener(args.output, 'wt') as fh_out:
        for rec in SeqIO.parse(fh_in, 'fasta'):
            if rec.id in ids:
                # Sanitize headers: phold/pharokka choke on colons
                if ':' in rec.id or ':' in rec.description:
                    rec.id = rec.id.replace(':', '_')
                    rec.description = rec.description.replace(':', '_')
                    rec.name = rec.name.replace(':', '_')
                    sanitized += 1
                SeqIO.write(rec, fh_out, 'fasta')
                written += 1

    print(f"Wrote {written}/{len(ids)} genomes to {args.output}")
    if sanitized:
        print(f"  ({sanitized} headers sanitized: ':' replaced with '_')")
    if written < len(ids):
        print(f"  ({len(ids) - written} IDs not found in FASTA)")


if __name__ == '__main__':
    main()
