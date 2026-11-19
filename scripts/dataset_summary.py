"""Dataset composition (the paper's Table 1): slides, patients and spectra per methylation class.

Slides, patients and raw spectra come from the raw cubes; retained spectra from a processed dataset.

Usage:
    python scripts/dataset_summary.py [--raw-dir data/raw] [--processed data/processed/glioma_mil.h5]
"""
import argparse
import collections
import re
from pathlib import Path

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default='data/raw')
    ap.add_argument('--processed', default='data/processed/glioma_mil.h5')
    args = ap.parse_args()

    slides = collections.defaultdict(lambda: {'slides': 0, 'patients': set(), 'raw': 0})
    for path in sorted(Path(args.raw_dir).rglob('*.npy')):
        cls = int(re.search(r'LGm-(\d)', str(path)).group(1))
        patient = re.search(r'HF-(\d+)_', path.name).group(1)
        rows, cols = np.load(path, mmap_mode='r').shape[:2]
        entry = slides[cls]
        entry['slides'] += 1
        entry['patients'].add(patient)
        entry['raw'] += rows * cols

    retained = {}
    if Path(args.processed).exists():
        with h5py.File(args.processed, 'r') as f:
            labels, counts = np.unique(f['y_bag'][:], return_counts=True)
        # y_bag is 0-indexed (LGm1 = 0)
        retained = {int(label) + 1: int(n) for label, n in zip(labels, counts)}

    print(f'{"Class":8s}{"Slides":>8s}{"Patients":>10s}{"Spectra":>10s}{"Retained":>10s}')
    for cls in sorted(slides):
        e = slides[cls]
        print(f'LGm{cls:<5d}{e["slides"]:8d}{len(e["patients"]):10d}{e["raw"]:10,d}{retained.get(cls, 0):10,d}')
    all_patients = set().union(*(e['patients'] for e in slides.values()))
    print(f'{"Total":8s}{sum(e["slides"] for e in slides.values()):8d}{len(all_patients):10d}'
          f'{sum(e["raw"] for e in slides.values()):10,d}{sum(retained.values()):10,d}')


if __name__ == '__main__':
    main()
