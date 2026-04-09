import numpy as np
import h5py
import os
import shutil
from pathlib import Path
from src.data.preprocessing import RamanPreprocessor
from src.data.loader import GliomaDataLoader

def test_isolated_pipeline():
    print("--- Setting up Isolated Test Environment ---")
    
    # Define temporary test paths
    test_base = Path("data/temp_test")
    raw_dir = test_base / "raw/LGm-1"
    map_path = test_base / "tumor_maps.npy"
    output_h5 = test_base / "processed/test_dataset.h5"
    
    # 1. Clean previous runs
    if test_base.exists():
        shutil.rmtree(test_base)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate Dummy Raw Data (Simulating a 10x10 scan with 1000 bands)
    print("Generating dummy raw spectra...")
    dummy_spectra = np.random.rand(10, 10, 1000).astype(np.float32)
    # Save as if it were a real patient file
    np.save(raw_dir / "HF-9999_Test_1.npy", dummy_spectra)
    
    # 3. Generate Matching Dummy Tumor Map
    # CRITICAL: The map must match the flattened shape of the spectra (10x10 = 100 pixels)
    print("Generating dummy tumor map...")
    dummy_map = np.random.randint(0, 2, size=(10, 10)) # 0 or 1 labels
    # The preprocessor expects a LIST or ARRAY of maps, one per sample
    # We only have 1 sample, so we save an array containing just this one map
    np.save(map_path, np.array([dummy_map], dtype=object))
    
    print("\n--- Testing Preprocessor ---")
    # Initialize processor pointing to the TEMP folders
    processor = RamanPreprocessor(
        raw_data_dir=test_base / "raw",
        tumor_maps_path=map_path,
        output_file=output_h5
    )
    
    # Run pipeline
    processor.run()
    
    print("\n--- Testing Loader ---")
    # Initialize loader on the generated H5 file
    loader = GliomaDataLoader(output_h5, memory_mapped=False)
    
    print(f"Loader Stats:")
    print(f"  X shape: {loader.X.shape} (Expected: 100, 1000)")
    print(f"  y_tumor shape: {loader.y_tumor.shape} (Expected: 100,)")
    print(f"  Sample IDs: {np.unique(loader.groups)}")

    print("\n--- Testing Group Splitting ---")
    # We need at least 2 samples to split properly, but let's see if it handles 1 gracefully
    # or just checks the generator exists.
    try:
        gen = loader.get_cross_validation_generator(n_splits=2)
        print("  Generator created successfully.")
    except ValueError as e:
        print(f"  (Expected warning) Splitter might complain with only 1 sample: {e}")

    # Optional: Cleanup
    # shutil.rmtree(test_base)
    print("\nTest Complete. You can inspect 'data/temp_test' if needed.")

if __name__ == "__main__":
    test_isolated_pipeline()