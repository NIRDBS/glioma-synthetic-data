"""
Stage 1: Preprocessing
Loads raw data, applies filters, outputs HDF5
"""

from pathlib import Path
from src.data.preprocessing import RamanPreprocessor


def preprocessing_stage(config, output_dir):
    """
    Execute preprocessing stage.
    
    Args:
        config: Configuration dict with preprocessing settings
        output_dir: Output directory for results
    
    Returns:
        Path to output HDF5 file
    """
    
    output_h5 = Path(config['paths']['processed_file'])
    print(f"\n{'='*80}")
    print(f"PREPROCESSING STAGE")
    print(f"{'='*80}")
    print(f"Input:  {config['paths']['raw_data_dir']}")
    print(f"Output: {output_h5}")
    
    # Initialize preprocessor with full config
    preprocessor = RamanPreprocessor(
        raw_data_dir=Path(config['paths']['raw_data_dir']),
        output_file=output_h5,
        tumor_maps_path=Path(config['preprocessing'].get('tumor_maps_path')),
        calibration_path=Path(config['paths']['calib_file']),
        mask_strategy=config['preprocessing'].get('mask_strategy', 'tumor_map'),
        config=config
    )
    
    # Run preprocessing
    print(f"\nConfiguration:")
    print(f"  • Spatial median filter: {config['preprocessing'].get('spatial_median_filter', False)}")
    print(f"  • Spatial smoothing: {config['preprocessing'].get('apply_spatial_smoothing', False)}")
    print(f"  • Quality filtering: {config['preprocessing'].get('apply_quality_filter', False)}")
    print(f"  • Cosmic ray removal: {config['preprocessing'].get('cosmic_ray_removal', False)}")
    print(f"  • Tumor mask source: {config['preprocessing'].get('tumor_maps_path', 'clustering')}")
    
    print(f"\nStarting preprocessing...")
    preprocessor.run()
    
    print(f"\n✓ Preprocessing complete")
    print(f"  Output file: {output_h5}")
    
    return output_h5
    print(f"  Output HDF5: {output_h5}")
    print(f"  Size: {output_h5.stat().st_size / (1024**3):.2f} GB")
    
    return output_h5
