"""
Build a standalone, pre-filtered + pre-labeled dataset for an intra-mutant
LGm subtype task (e.g. exp_671: LGm1 vs {LGm2, LGm3}), WITHOUT touching any
shared pipeline code.

Input: data/processed/glioma_mil.h5 (the existing v1 dataset; its y_bag is
already the 0-indexed 6-class LGm1..6 ground truth, parsed from the raw
LGm-<k>/HF-<n>_... folder names -- see src/data/preprocessing.py:49).

What this script does:
  1. Keeps only instances whose class_id is in {0, 1, 2} (LGm1/2/3, the
     IDH-mutant population) -- LGm4-6 rows are dropped entirely, not merged
     into a negative class.
  2. Re-encodes y_bag so the EXISTING, unmodified nc==2 binary rule in
     pipeline/_common.py (``np.where(y < 3, 0, 1)``) produces the correct
     final 0/1 label with zero code changes anywhere downstream:
       - the designated --positive-class is written as 3 (an arbitrary
         value >= 3, so the existing rule's "1" branch fires)
       - the other two classes are written as 0 (an arbitrary value < 3,
         so the existing rule's "0" branch fires)
     The output file's y_bag therefore only ever contains {0, 3}, NOT the
     more intuitive {0, 1} -- this is deliberate, documented here and in
     the file's own HDF5 attrs, not a bug.
  3. Every other field (coords, patient_ids, sample_ids, spectra,
     wavenumbers, tissue_cluster, viable_legacy) is subset by the same
     instance mask; slide_names/slide_shapes are subset by the
     corresponding slide mask.

Usage:
    python scripts/build_glioma_mil_lgm_subtype.py --positive-class 0 \
        --out data/processed/glioma_mil_lgm1_vs_rest.h5
    python scripts/build_glioma_mil_lgm_subtype.py --positive-class 1 \
        --out data/processed/glioma_mil_lgm2_vs_rest.h5
    python scripts/build_glioma_mil_lgm_subtype.py --positive-class 2 \
        --out data/processed/glioma_mil_lgm3_vs_rest.h5
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_CLASSES = (0, 1, 2)  # LGm1, LGm2, LGm3 -- the IDH-mutant population
LGM_NAMES = {0: "LGm1", 1: "LGm2", 2: "LGm3"}


def build(in_path, out_path, positive_class):
    if positive_class not in INCLUDE_CLASSES:
        raise ValueError(f"positive_class must be one of {INCLUDE_CLASSES}, got {positive_class}")
    rest = [c for c in INCLUDE_CLASSES if c != positive_class]

    with h5py.File(in_path, 'r') as f:
        y_bag = f['y_bag'][:]
        if y_bag.size and y_bag.min() == 1:
            y_bag = y_bag - 1  # defensive, matches pipeline/_common.py's own fix

        inst_mask = np.isin(y_bag, INCLUDE_CLASSES)
        n_before, n_after = y_bag.size, int(inst_mask.sum())

        sample_ids = f['sample_ids'][:].astype(str)
        slide_names = f['slide_names'][:].astype(str)
        kept_sample_ids = set(sample_ids[inst_mask])
        slide_mask = np.isin(slide_names, list(kept_sample_ids))

        # remap: positive_class -> 3 (existing rule's ">=3" branch -> final 1)
        #        other included classes -> 0 (existing rule's "<3" branch -> final 0)
        y_remapped = np.where(y_bag[inst_mask] == positive_class, 3, 0).astype(np.int32)

        bag_counts_before = {int(c): int((y_bag[np.unique(sample_ids, return_index=True)[1]] == c).sum())
                              for c in INCLUDE_CLASSES}

        with h5py.File(out_path, 'w') as out:
            out.create_dataset('spectra', data=f['spectra'][:][inst_mask])
            out.create_dataset('coords', data=f['coords'][:][inst_mask])
            out.create_dataset('patient_ids', data=f['patient_ids'][:][inst_mask])
            out.create_dataset('sample_ids', data=f['sample_ids'][:][inst_mask])
            out.create_dataset('tissue_cluster', data=f['tissue_cluster'][:][inst_mask])
            out.create_dataset('viable_legacy', data=f['viable_legacy'][:][inst_mask])
            out.create_dataset('wavenumbers', data=f['wavenumbers'][:])
            out.create_dataset('y_bag', data=y_remapped)
            out.create_dataset('slide_names', data=f['slide_names'][:][slide_mask])
            out.create_dataset('slide_shapes', data=f['slide_shapes'][:][slide_mask])

            out.attrs['schema_version'] = 'mil-v1-lgm-subtype'
            out.attrs['source_file'] = str(in_path)
            out.attrs['label_scheme'] = (
                f"{LGM_NAMES[positive_class]} vs rest, within IDH-mutant "
                f"({'/'.join(LGM_NAMES[c] for c in INCLUDE_CLASSES)}) only"
            )
            out.attrs['positive_class_raw_lgm'] = LGM_NAMES[positive_class]
            out.attrs['rest_classes_raw_lgm'] = '/'.join(LGM_NAMES[c] for c in rest)
            out.attrs['y_bag_encoding'] = (
                "y_bag in this file is {0, 3}, NOT {0, 1}: 3 marks the positive "
                "class, 0 marks the rest -- chosen so the existing, UNMODIFIED "
                "nc==2 rule in pipeline/_common.py (np.where(y<3,0,1)) produces "
                "the correct final binary label with zero code changes. See "
                "scripts/build_glioma_mil_lgm_subtype.py for the full rationale."
            )
            out.attrs['n_instances'] = n_after
            out.attrs['n_slides'] = int(slide_mask.sum())
            out.attrs['baseline_method'] = f.attrs.get('baseline_method', 'iarpls')
            out.attrs['baseline_lambda'] = f.attrs.get('baseline_lambda', 10000.0)
            out.attrs['median_kernel'] = f.attrs.get('median_kernel', 3)

    print(f"Source: {in_path}  ({n_before} instances, 59 slides)")
    print(f"Kept (LGm1/2/3 only): {n_after} instances, {int(slide_mask.sum())} slides")
    print(f"Per-bag class counts in source population: {bag_counts_before}")
    print(f"Positive class: {LGM_NAMES[positive_class]} ({bag_counts_before[positive_class]} bags)  "
          f"vs rest ({sum(v for k, v in bag_counts_before.items() if k != positive_class)} bags)")
    print(f"Written -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in-path', default=str(PROJECT_ROOT / 'data/processed/glioma_mil.h5'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--positive-class', type=int, required=True, choices=[0, 1, 2],
                         help="0=LGm1, 1=LGm2, 2=LGm3 (0-indexed, 'vs rest' within LGm1/2/3)")
    args = parser.parse_args()
    build(args.in_path, args.out, args.positive_class)
