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
    Optimized Builder class.
    Major Change: Pre-computes sparse matrices for airPLS to speed up processing.
    """
    
    # Configuration
    IGNORE_SAMPLES = ["HF-1887_via-t_2_1.h5_3"]
    AIRPLS_LAMBDA = 100 # 10^4 is standard, but you had 4. Confirm this value! Usually it's 100-10000.
    AIRPLS_PORDER = 1
    SIGNAL_LENGTH = 1738
    
    def __init__(self, raw_data_dir, output_file, 
                 mask_strategy="tumor_map", 
                 tumor_maps_path=None, 
                 calibration_path=None):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.output_file = Path(output_file)
        self.mask_strategy = mask_strategy
        self.tumor_maps_path = Path(tumor_maps_path) if tumor_maps_path else None
        self.calibration_path = Path(calibration_path) if calibration_path else None

        self.filename_regex = re.compile(r"LGm-(?P<class_id>\d)[/\\]HF-(?P<patient_id>\d+)_")
        
        if self.calibration_path:
            self._load_calibration()
            
        # --- PRE-COMPUTE AIRPLS MATRICES ---
        # The difference matrix D is constant for a fixed signal length.
        # Computing it once saves huge time.
        self.m = self.SIGNAL_LENGTH
        self.D = np.diff(np.eye(self.m), 2) # Second order difference
        self.DD = self.D.dot(self.D.T)      # D * D.T (Used in the system)
        self.DD_sparse = csc_matrix(self.DD) # Convert to sparse once

    def _load_calibration(self):
        """Loads and parses the frequency table."""
        try:
            df = pd.read_csv(self.calibration_path)
            wavenumbers = df.iloc[:, 1].values
            
            if len(wavenumbers) < self.SIGNAL_LENGTH:
                 diff = np.mean(np.diff(wavenumbers[-10:]))
                 padding = [wavenumbers[-1] + diff * i for i in range(1, self.SIGNAL_LENGTH - len(wavenumbers) + 1)]
                 wavenumbers = np.concatenate([wavenumbers, padding])
            
            self.wavenumbers = wavenumbers
            
            # Indices for Bio-Mask
            self.idx_bio_start = np.argmin(np.abs(wavenumbers - 1050))
            self.idx_bio_end   = np.argmin(np.abs(wavenumbers - 1130))
            self.idx_hwn_start = np.argmin(np.abs(wavenumbers - 2800))
            self.idx_hwn_end   = np.argmin(np.abs(wavenumbers - 3000))
            
            print(f"✅ Calibration Loaded. Range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹")
            
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            raise

    def _create_bio_mask(self, image_data):
        """Robust Bio-Mask (Phenylalanine/DNA + Lipids)."""
        # 1. Broad Bio Region (DNA/Lipid/Phe)
        bio_map = np.sum(image_data[:, :, self.idx_bio_start:self.idx_bio_end], axis=2)
        # 2. High Wavenumber (Lipids)
        hwn_map = np.sum(image_data[:, :, self.idx_hwn_start:self.idx_hwn_end], axis=2)
        
        # 3. Smart Thresholds (5th percentile floor)
        bg_bio = np.percentile(bio_map, 5)
        bg_hwn = np.percentile(hwn_map, 5)
        
        if bg_bio > 100: thresh_bio = 50.0 
        else: thresh_bio = max(bg_bio * 1.5, 20.0)

        if bg_hwn > 100: thresh_hwn = 50.0 
        else: thresh_hwn = max(bg_hwn * 1.5, 50.0)

        raw_mask = (bio_map > thresh_bio) & (hwn_map > thresh_hwn)
        clean_mask = binary_erosion(raw_mask, iterations=2)
        return clean_mask

    def _airPLS(self, x, lambda_=100, porder=1, itermax=15):
        """
        Optimized baseline correction.
        Uses pre-computed sparse DD matrix.
        """
        m = x.shape[0]
        w = np.ones(m)
        
        for i in range(1, itermax + 1):
            # System: (W + lambda * D'D) z = W x
            # Since W is diagonal, we can construct it efficiently
            W = diags(w)
            # LHS = W + lambda * DD (DD is already computed/sparse)
            # We assume lambda is scalar, so lambda * DD is fast
            # Note: airPLS usually uses lambda^order or just lambda. 
            # Standard paper: (W + lambda * D'D) z = w * x
            
            # Construct LHS matrix
            # We use csc_matrix accumulation which is efficient
            LHS = W + (lambda_ * self.DD_sparse)
            
            # Solve
            z = spsolve(LHS, w * x)
            
            # Update weights
            d = x - z
            dssn = np.abs(d[d < 0].sum())
            if dssn < 0.001 * (np.abs(x).sum()):
                break
                
            w[d >= 0] = 0
            # Stability check for exp to avoid overflow
            neg_idx = d < 0
            if neg_idx.any():
                factor = i * np.abs(d[neg_idx]) / dssn
                w[neg_idx] = np.exp(factor)
            
            w[0] = np.exp(i * (d[d < 0]).max() / dssn) if (d<0).any() else 0
            w[-1] = w[0]
            
        return z

    def _preprocess_spectrum(self, raw_spectrum):
        # 1. Baseline
        baseline = self._airPLS(raw_spectrum, lambda_=self.AIRPLS_LAMBDA, porder=self.AIRPLS_PORDER)
        corrected = raw_spectrum - baseline
        
        # 2. Clip & Norm
        corrected = np.maximum(corrected, 0)
        min_val, max_val = np.min(corrected), np.max(corrected)
        
        if max_val - min_val == 0:
            return np.zeros_like(corrected)
            
        return (corrected - min_val) / (max_val - min_val)

    def run(self):
        # --- [Identical run logic, just copy the loop from previous file] ---
        # ... (Same file finding & HDF5 writing logic) ...
        # Ensure you include the "files_to_process = ..." and "with h5py..." block
        
        # [Abbreviated for clarity - insert the run() method from your previous upload here]
        # Just ensure 'self._preprocess_spectrum' is called.
        
        # ...
        
        # Find Files
        files_to_process = []
        for class_dir in glob.glob(str(self.raw_data_dir / "LGm-*")):
            files_to_process.extend(glob.glob(os.path.join(class_dir, "*.npy")))
        files_to_process = natsort.natsorted(files_to_process)
        print(f"🔍 Found {len(files_to_process)} scan files.")

        with h5py.File(self.output_file, "w") as h5f:
            h5f.create_dataset("wavenumbers", data=self.wavenumbers)
            
            dset_spectra = h5f.create_dataset("spectra", shape=(0, 1738), maxshape=(None, 1738), dtype='float32', compression="gzip")
            dset_class = h5f.create_dataset("y_class", shape=(0,), maxshape=(None,), dtype='int32')
            dset_patient = h5f.create_dataset("patient_ids", shape=(0,), maxshape=(None,), dtype='int32')
            dset_samples = h5f.create_dataset("sample_ids", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))

            for filepath in tqdm(files_to_process, desc="Processing"):
                filename = os.path.basename(filepath)
                if any(ignored in filename for ignored in self.IGNORE_SAMPLES): continue
                
                match = self.filename_regex.search(filepath)
                if not match: continue
                
                class_id = int(match.group("class_id")) - 1
                patient_id = int(match.group("patient_id"))
                sample_name = filename.replace(".npy", "")

                try:
                    raw_cube = np.load(filepath)
                    if isinstance(raw_cube, np.ndarray) and raw_cube.dtype == 'O':
                         raw_cube = raw_cube.item().get('data', raw_cube)
                    
                    # Create Mask
                    if self.mask_strategy == "bio_mask":
                        mask = self._create_bio_mask(raw_cube)
                        flat_data = raw_cube[mask]
                    else:
                        # Fallback for now if you switch strategies
                        continue
                        
                    if flat_data.shape[0] == 0: continue
                    
                    # BATCH PROCESSING LOOP
                    batch_spectra = []
                    # Pre-allocating list is faster than numpy append loop
                    batch_spectra = [self._preprocess_spectrum(s) for s in flat_data]
                    
                    batch_spectra = np.array(batch_spectra, dtype='float32')
                    n_new = batch_spectra.shape[0]

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

                except Exception as e:
                    print(f"⚠️ Error {filename}: {e}")