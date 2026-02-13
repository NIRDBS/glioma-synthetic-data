import os
import glob
import re
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.ndimage import binary_erosion
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import natsort

class RamanPreprocessor:
    """
    Builder class that reads raw .npy files, applies airPLS baseline correction,
    normalization, and saves the result to a single HDF5 file.
    
    Supports two masking strategies:
    1. 'tumor_map': Uses pre-annotated masks from the dataset authors (Ground Truth).
    2. 'bio_mask':  Generates a fresh mask using biological heuristics (Phe + Lipids).
    """
    
    # Configuration
    IGNORE_SAMPLES = ["HF-1887_via-t_2_1.h5_3"]
    AIRPLS_LAMBDA = 4
    AIRPLS_PORDER = 1
    
    def __init__(self, raw_data_dir, output_file, 
                 mask_strategy="tumor_map", 
                 tumor_maps_path=None, 
                 calibration_path=None):
        """
        args:
            raw_data_dir: Path to 'data/raw' containing LGm-* folders.
            output_file: Path where the processed .h5 file will be saved.
            mask_strategy: "tumor_map" or "bio_mask".
            tumor_maps_path: Path to 'tumor_maps.npy' (Required for 'tumor_map').
            calibration_path: Path to 'RamanFreqTable...csv' (Required for both strategies).
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_file = Path(output_file)
        self.mask_strategy = mask_strategy
        
        # Calibration path is required for all strategies (to get wavenumbers for output)
        if not calibration_path:
            raise ValueError("calibration_path is required")
        self.calibration_path = Path(calibration_path)
        
        # Strategy-specific validation
        if self.mask_strategy == "tumor_map":
            if not tumor_maps_path:
                raise ValueError("tumor_maps_path is required for 'tumor_map' strategy")
            self.tumor_maps_path = Path(tumor_maps_path)
            
        elif self.mask_strategy == "bio_mask":
            self.tumor_maps_path = None
            
        else:
            raise ValueError(f"Unknown mask strategy: {mask_strategy}")

        # Load Calibration (always needed)
        self._load_calibration()
        
        self.filename_regex = re.compile(r"LGm-(?P<class_id>\d)[/\\]HF-(?P<patient_id>\d+)_")

    def _load_calibration(self):
        """Loads and parses the frequency table to find key indices for Bio-Mask."""
        try:
            df = pd.read_csv(self.calibration_path)
            wavenumbers = df['freq'].values
            
            # Handle length mismatch (extrapolate to 1738)
            if len(wavenumbers) < 1738:
                 diff = np.mean(np.diff(wavenumbers[-10:]))
                 padding = [wavenumbers[-1] + diff * i for i in range(1, 1738 - len(wavenumbers) + 1)]
                 wavenumbers = np.concatenate([wavenumbers, padding])
            
            self.wavenumbers = wavenumbers
            
            # Pre-calculate indices for efficiency
            self.idx_phe_start = np.argmin(np.abs(wavenumbers - 1000)) # Phenylalanine
            self.idx_phe_end = np.argmin(np.abs(wavenumbers - 1050))
            self.idx_hwn_start = np.argmin(np.abs(wavenumbers - 2850)) # Lipids Start
            self.idx_hwn_end = np.argmin(np.abs(wavenumbers - 2950))   # Lipids End
            self.idx_bio_start = np.argmin(np.abs(self.wavenumbers - 1050))
            self.idx_bio_end   = np.argmin(np.abs(self.wavenumbers - 1130))
        
            print(f"✅ Calibration Loaded. Phe Range: {self.idx_phe_start} - {self.idx_phe_end}, HWN Range: {self.idx_hwn_start} - {self.idx_hwn_end}")
            
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            raise

    def _create_bio_mask(self, image_data):
        """
        Generates a mask for BIOLOGICAL TISSUE.
        Robust to 'Full Coverage' scans where tissue fills the frame.
        """
        # 1. Primary Signal: Lipid/DNA/Phosphate Region (Broad & Strong)
        # Replaces the weak Phenylalanine (1004) check
        idx_bio_start = np.argmin(np.abs(self.wavenumbers - 1050))
        idx_bio_end   = np.argmin(np.abs(self.wavenumbers - 1130))
        bio_map = np.sum(image_data[:, :, idx_bio_start:idx_bio_end], axis=2)
        
        # 2. Validation Signal: C-H Stretch (High Wavenumber) - The "Green" Channel
        hwn_map = np.sum(image_data[:, :, self.idx_hwn_start:self.idx_hwn_end], axis=2)
        
        # 3. Smart Thresholding (The Fix)
        # Use 5th percentile to find the true dark spots (glass)
        bg_bio = np.percentile(bio_map, 5)
        bg_hwn = np.percentile(hwn_map, 5)
        
        # SATURATION CHECK: If the 'background' is bright, the image is 100% tissue.
        # We switch to an absolute floor to avoid rejecting good tissue.
        # (Values based on typical Raman counts; 100 is a safe 'signal' floor)
        
        if bg_bio > 100: 
            thresh_bio = 50.0 # Absolute floor
        else:
            thresh_bio = max(bg_bio * 1.5, 20.0) # Standard relative threshold

        if bg_hwn > 100:
            thresh_hwn = 50.0 # Absolute floor
        else:
            thresh_hwn = max(bg_hwn * 1.5, 50.0)

        # 4. The Logic: Must have BOTH (prevents Glass false positives)
        raw_mask = (bio_map > thresh_bio) & (hwn_map > thresh_hwn)
        
        # 5. Clean up
        clean_mask = binary_erosion(raw_mask, iterations=2)
        
        return clean_mask

    def _airPLS(self, x, lambda_, porder, itermax=15):
        """Baseline correction algorithm."""
        m = x.shape[0]
        w = np.ones(m)
        for i in range(1, itermax + 1):
            z = spsolve(
                csc_matrix(np.diff(np.eye(m), 2).dot(np.diff(np.eye(m), 2).T) + np.diag(w * lambda_)),
                (w * x)
            )
            d = x - z
            dssn = np.abs(d[d < 0].sum())
            if dssn < 0.001 * (np.abs(x).sum()):
                break
            w[d >= 0] = 0
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
            w[0] = np.exp(i * (d[d < 0]).max() / dssn)
            w[-1] = w[0]
        return z

    def _preprocess_spectrum(self, raw_spectrum):
        """Applies airPLS baseline correction and MinMax normalization."""
        # 1. Baseline Correction
        baseline = self._airPLS(raw_spectrum, self.AIRPLS_LAMBDA, self.AIRPLS_PORDER)
        corrected = raw_spectrum - baseline
        
        # 2. Clip Negatives
        corrected = np.maximum(corrected, 0)
        
        # 3. Min-Max Normalization
        min_val = np.min(corrected)
        max_val = np.max(corrected)
        if max_val - min_val == 0:
            return np.zeros_like(corrected)
        
        return (corrected - min_val) / (max_val - min_val)

    def _build_file_to_map_correspondence(self, files_to_process, tumor_maps_array):
        """
        Builds correspondence by shape-matching with greedy assignment.
        This ensures no dimension mismatch errors will occur.
        """
        file_to_map = {}
        used_map_indices = set()
        
        print("🔗 Building file-to-map correspondence by shape matching...")
        
        for filepath in files_to_process:
            filename = os.path.basename(filepath)
            
            try:
                raw_cube = np.load(filepath)
                if isinstance(raw_cube, np.ndarray) and raw_cube.dtype == 'O':
                    raw_cube = raw_cube.item().get('data', raw_cube)
                
                cube_h, cube_w = raw_cube.shape[:2]
                
                # Find the first unused map with matching dimensions
                matched_map = None
                matched_idx = None
                
                for map_idx, t_map in enumerate(tumor_maps_array):
                    if map_idx in used_map_indices:
                        continue
                    map_h, map_w = t_map.shape[:2]
                    
                    if cube_h == map_h and cube_w == map_w:
                        matched_map = t_map
                        matched_idx = map_idx
                        break
                
                if matched_map is not None:
                    file_to_map[filepath] = matched_map
                    used_map_indices.add(matched_idx)
                else:
                    file_to_map[filepath] = None
                    if "HF-9999" not in filename:
                        print(f"  ⚠️  {filename:40s} ({cube_h:3d}x{cube_w:3d}) -> NO MATCHING MAP")
                    
            except Exception as e:
                print(f"  ❌ {filename}: {e}")
                file_to_map[filepath] = None
        
        matched_count = sum(1 for v in file_to_map.values() if v is not None)
        print(f"✅ Matched {matched_count}/{len(files_to_process)} files to maps\n")
        
        return file_to_map

    def run(self):
        """Executes the pipeline."""
        
        # --- 1. Find Files ---
        files_to_process = []
        for class_dir in glob.glob(str(self.raw_data_dir / "LGm-*")):
            files_to_process.extend(glob.glob(os.path.join(class_dir, "*.npy")))
        
        files_to_process = natsort.natsorted(files_to_process)
        print(f"🔍 Found {len(files_to_process)} scan files.")
        
        # --- 2. Load Resources based on Strategy ---
        file_to_map = {}
        if self.mask_strategy == "tumor_map":
            print(f"📂 Loading Tumor Maps from {self.tumor_maps_path}...")
            try:
                tumor_maps_data = np.load(self.tumor_maps_path, allow_pickle=True)
                
                # Check if it's a dict or array
                if isinstance(tumor_maps_data, np.ndarray) and tumor_maps_data.dtype == object:
                    # Try to unpack as dict if size is 1
                    if tumor_maps_data.size == 1:
                        tumor_maps_array = tumor_maps_data.item()
                        if not isinstance(tumor_maps_array, dict):
                            raise ValueError("Expected dict after unpacking single-element array")
                    else:
                        # It's an array of maps - need to match by shape
                        tumor_maps_array = tumor_maps_data
                elif isinstance(tumor_maps_data, dict):
                    tumor_maps_array = tumor_maps_data
                else:
                    raise ValueError(f"Unexpected tumor_maps type: {type(tumor_maps_data)}")
                
                map_count = len(tumor_maps_array) if isinstance(tumor_maps_array, (dict, np.ndarray)) else 0
                print(f"✅ Loaded {map_count} maps.")
                
                # Build file-to-map correspondence
                if isinstance(tumor_maps_array, np.ndarray) and tumor_maps_array.dtype == object:
                    file_to_map = self._build_file_to_map_correspondence(files_to_process, tumor_maps_array)
                else:
                    # Dict format - keep simple mapping
                    file_to_map = {f: tumor_maps_array.get(os.path.basename(f)) for f in files_to_process}
                    
            except Exception as e:
                print(f"❌ Error loading tumor maps: {e}")
                return

        # --- 3. Process & Stream to HDF5 ---
        with h5py.File(self.output_file, "w") as h5f:
            # --- Save Wavenumbers for Plots ---
            h5f.create_dataset("wavenumbers", data=self.wavenumbers)
            print("✅ Saved 'wavenumbers' vector to HDF5.")
            
            # Initialize resizable datasets
            dset_spectra = h5f.create_dataset("spectra", shape=(0, 1738), maxshape=(None, 1738), dtype='float32', compression="gzip")
            dset_class = h5f.create_dataset("y_class", shape=(0,), maxshape=(None,), dtype='int32')
            dset_patient = h5f.create_dataset("patient_ids", shape=(0,), maxshape=(None,), dtype='int32')
            dset_samples = h5f.create_dataset("sample_ids", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))

            for file_num, filepath in enumerate(tqdm(files_to_process, desc="Processing")):
                filename = os.path.basename(filepath)
                
                # Skip excluded samples
                if any(ignored in filename for ignored in self.IGNORE_SAMPLES):
                    continue

                # Parse Metadata
                match = self.filename_regex.search(filepath)
                if not match:
                    continue
                
                class_id = int(match.group("class_id")) - 1 # Shift 1-6 -> 0-5
                patient_id = int(match.group("patient_id"))
                sample_name = filename.replace(".npy", "")

                # Load Data
                try:
                    raw_cube = np.load(filepath)
                    # Handle Apollo structure if it's a dict inside npy
                    if isinstance(raw_cube, np.ndarray) and raw_cube.dtype == 'O':
                         raw_cube = raw_cube.item().get('data', raw_cube)
                except Exception as e:
                    print(f"⚠️ Corrupt file {filename}: {e}")
                    continue

                # --- MASKING LOGIC ---
                valid_mask = None
                
                if self.mask_strategy == "tumor_map":
                    # Get map from correspondence dict
                    t_map = file_to_map.get(filepath)
                    
                    if t_map is not None:
                        # Apollo Map: 1=Tumor, 0=Non-Tumor(Glass/Normal)
                        # We only keep 1 (Tumor)
                        valid_mask = (t_map == 1)
                    else:
                        # Skip if no map found
                        continue
                        
                elif self.mask_strategy == "bio_mask":
                    # Generate mask from physics
                    valid_mask = self._create_bio_mask(raw_cube)

                # Flatten Valid Pixels (TISSUE ONLY)
                if valid_mask is not None:
                    # Ensure mask shape matches image (H, W)
                    if valid_mask.shape != raw_cube.shape[:2]:
                        print(f"⚠️ Dimension mismatch {filename}: Data {raw_cube.shape} vs Mask {valid_mask.shape}")
                        continue
                        
                    # Extract only tissue pixels
                    flat_data = raw_cube[valid_mask]
                else:
                    continue

                if flat_data.shape[0] == 0:
                    continue

                # Preprocess Batch (ONLY TISSUE PIXELS - much faster!)
                n_spectra = flat_data.shape[0]
                print(f"[{file_num}] {filename}: {n_spectra} tissue spectra", flush=True)
                
                batch_spectra = []
                for i in range(n_spectra):
                    batch_spectra.append(self._preprocess_spectrum(flat_data[i]))
                    if (i + 1) % 2000 == 0:
                        print(f"  ... {i+1}/{n_spectra} spectra processed", flush=True)
                
                batch_spectra = np.array(batch_spectra, dtype='float32')
                n_new = batch_spectra.shape[0]

                # Append to HDF5
                current_size = dset_spectra.shape[0]
                new_size = current_size + n_new
                
                dset_spectra.resize(new_size, axis=0)
                dset_class.resize(new_size, axis=0)
                dset_patient.resize(new_size, axis=0)
                dset_samples.resize(new_size, axis=0)
                
                dset_spectra[current_size:] = batch_spectra
                dset_class[current_size:] = np.full(n_new, class_id, dtype='int32')
                dset_patient[current_size:] = np.full(n_new, patient_id, dtype='int32')
                dset_samples[current_size:] = np.full(n_new, sample_name)
                print(f"  ✅ Saved {n_new} spectra. Total: {new_size}", flush=True)

        print(f"✅ Processing complete. Saved to {self.output_file}")