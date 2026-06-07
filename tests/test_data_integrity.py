import numpy as np
import glob
import os
import natsort
from pathlib import Path

# Configuration
RAW_DIR = "data/raw"
MAP_PATH = "data/tumor_maps.npy"
IGNORE_SAMPLES = ["HF-1887_via-t_2_1.h5_3"]

def test_alignment_and_shapes():
    print(f"--- Diagnostic: Checking Alignment between {RAW_DIR} and {MAP_PATH} ---")
    
    # 1. Load Maps
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"Tumor maps not found at {MAP_PATH}")
    
    tumor_maps = np.load(MAP_PATH, allow_pickle=True)
    print(f"Loaded {len(tumor_maps)} tumor maps.")

    # 2. Load Files (Strategy: Sort by Filename Stem)
    all_files = glob.glob(f"{RAW_DIR}/LGm*/*.npy")
    # Sort by the filename (e.g., HF-448) to match standard research workflows
    sorted_files = natsort.natsorted(all_files, key=lambda x: Path(x).stem)
    
    print(f"Found {len(sorted_files)} raw files.")
    
    # 3. Validation Loop
    mismatches = []
    
    # We iterate assuming 1-to-1 correspondence BEFORE filtering
    limit = min(len(tumor_maps), len(sorted_files))
    
    print(f"\nChecking first {limit} pairs...")
    
    for i in range(limit):
        file_path = sorted_files[i]
        t_map = tumor_maps[i]
        sample_name = Path(file_path).stem
        
        # Quick load to check shape (mmap_mode='r' is instant, reads header only)
        try:
            raw_data = np.load(file_path, mmap_mode='r')
            
            # Calculate expected flat size
            if raw_data.ndim == 3:
                flat_size = raw_data.shape[0] * raw_data.shape[1]
                dims = f"3D {raw_data.shape}"
            else:
                flat_size = raw_data.shape[0]
                dims = f"2D {raw_data.shape}"
                
            map_size = t_map.size if t_map is not None else 0
            
            status = " OK"
            if flat_size != map_size:
                status = " MISMATCH"
                mismatches.append((i, sample_name, flat_size, map_size))
            
            # Check if this is the ignored sample
            is_ignored = "(IGNORED)" if sample_name in IGNORE_SAMPLES else ""
            
            print(f"{i:02d} | {sample_name:<25} | File: {flat_size:<8} | Map: {map_size:<8} | {status} {is_ignored}")
            
        except Exception as e:
            print(f"{i:02d} | {sample_name:<25} | Error reading file: {e}")

    print("\n--- Summary ---")
    if len(mismatches) == 0:
        print("SUCCESS: All file shapes align with map shapes!")
        print("Guidance: You can proceed with processing using 'natsort' on filenames.")
    else:
        print(f"FAILURE: Found {len(mismatches)} shape mismatches.")
        print("Possible causes:")
        print("1. The sort order is different (try sorting by full path?).")
        print("2. 'tumor_maps.npy' might exclude the ignored sample, shifting indices.")
        print("3. 'tumor_maps.npy' might be sorted differently.")

if __name__ == "__main__":
    test_alignment_and_shapes()