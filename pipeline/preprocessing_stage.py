"""
Stage: Preprocessing.
Builds the modelling HDF5 from raw cubes plus the precomputed per-slide
clustering artifacts. See src/data/preprocessing.py for what this does and,
more importantly, what it deliberately no longer does (no mask_strategy
branch, no RamanSPy path, no re-running NMF/GMM clustering).
"""

from pathlib import Path

from src.data.preprocessing import RamanPreprocessor
from src.preprocessing.baseline import IARPLS_LAMBDA


def preprocessing_stage(config, output_dir=None):
    """
    Execute the preprocessing stage: raw cubes -> modelling HDF5.

    Args:
        config: experiment config dict. Reads paths.raw_data_dir, paths.calib_file,
            paths.processed_file (output) and, optionally, paths.processed_dir
            (input; defaults to '<paths.data_root>/processed') and
            preprocessing.baseline_lambda (defaults to the Stage 0 provisional
            constant, src.preprocessing.baseline.IARPLS_LAMBDA).
        output_dir: unused, accepted for call-signature parity with the other
            pipeline stages.

    Returns:
        Path to the written HDF5 file.
    """
    paths = config['paths']
    output_h5 = Path(paths['processed_file'])
    processed_dir = Path(paths.get('processed_dir', f"{paths.get('data_root', 'data')}/processed"))
    baseline_lambda = config.get('preprocessing', {}).get('baseline_lambda', IARPLS_LAMBDA)

    print(f"\n{'='*80}\nPREPROCESSING STAGE\n{'='*80}")
    print(f"Raw data:        {paths['raw_data_dir']}")
    print(f"Cluster artifacts: {processed_dir}  (must already exist per slide)")
    print(f"Output:          {output_h5}")
    print(f"Baseline:        IARPLS, lambda={baseline_lambda:g} (provisional -- "
          f"docs/stage0_findings.md 7.1)")

    preprocessor = RamanPreprocessor(
        raw_data_dir=paths['raw_data_dir'],
        processed_dir=processed_dir,
        output_file=output_h5,
        calibration_path=paths['calib_file'],
        baseline_lambda=baseline_lambda,
    )
    preprocessor.run()

    print(f"\nPreprocessing complete: {output_h5}")
    return output_h5
