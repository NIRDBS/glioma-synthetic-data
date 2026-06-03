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
import ramanspy as rp


class RamanPreprocessor:
    """
    Optimized Builder class.
    Major Change: Pre-computes sparse matrices for airPLS to speed up processing.
    """
    
    # Configuration
    IGNORE_SAMPLES = ["HF-1887_via-t_2_1.h5_3"]
    AIRPLS_LAMBDA = 100 
    AIRPLS_PORDER = 1
    SIGNAL_LENGTH = 1738
    
    def __init__(self, raw_data_dir, output_file, 
                 mask_strategy="tumor_map", 
                 tumor_maps_path=None, 
                 calibration_path=None,
                 use_ramanspy=False):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.output_file = Path(output_file)
        self.mask_strategy = mask_strategy
        self.tumor_maps_path = Path(tumor_maps_path) if tumor_maps_path else None
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.use_ramanspy = use_ramanspy
        self.CROP_REGION = (700, 1800)

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
            
            print(f" Calibration Loaded. Range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹")
            
        except Exception as e:
            print(f" Failed to load calibration: {e}")
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
    
    def _preprocess_batch_ramanspy(self, spectra_batch, wavenumbers_axis):
        """
        Applies Crop -> Denoise -> Baseline (IARPLS) -> Normalize using RamanSPy.
        """
        # 1. Wrap in Container
        container = rp.SpectralContainer(spectra_batch, wavenumbers_axis)
        
        # 2. Define Pipeline
        pipeline = rp.preprocessing.Pipeline([
            rp.preprocessing.misc.Cropper(region=self.CROP_REGION),
            rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
            rp.preprocessing.baseline.IARPLS(lam=10000, p=0.001), 
            rp.preprocessing.normalise.MinMax(pixelwise=True)
        ])
        
        # 3. Apply
        processed_container = pipeline.apply(container)
        return processed_container.spectral_data, processed_container.spectral_axis

    def run(self):
        """
        Executes the full preprocessing pipeline.
        
        MODES:
        1. Legacy Mode (use_ramanspy=False):
           - Keeps original 1738 spectral points.
           - Uses manual AirPLS and slow row-by-row processing.
           - Saves 'spectra' and 'spectra_smoothed'.
           
        2. RamanSPy Mode (use_ramanspy=True):
           - CROPS data to Fingerprint region (700-1800 cm-1).
           - Uses fast batch processing (IARPLS + SavGol).
           - Saves 'spectra' only (optimized size).
        """
        files_to_process = []
        for class_dir in glob.glob(str(self.raw_data_dir / "LGm-*")):
            files_to_process.extend(glob.glob(os.path.join(class_dir, "*.npy")))
        files_to_process = natsort.natsorted(files_to_process)
        print(f" Found {len(files_to_process)} scan files.")
        
        # --- 1. Load Tumor Maps (If needed) ---
        file_to_map = {}
        if self.mask_strategy == "tumor_map":
            print(f" Loading Tumor Maps from {self.tumor_maps_path}...")
            try:
                tumor_maps_data = np.load(self.tumor_maps_path, allow_pickle=True)
                if isinstance(tumor_maps_data, np.ndarray) and tumor_maps_data.dtype == object:
                     if tumor_maps_data.size == 1: tumor_maps_array = tumor_maps_data.item()
                     else: tumor_maps_array = tumor_maps_data
                else: tumor_maps_array = tumor_maps_data

                if isinstance(tumor_maps_array, (list, np.ndarray)):
                    file_to_map = self._build_file_to_map_correspondence(files_to_process, tumor_maps_array)
                else:
                    file_to_map = {f: tumor_maps_array.get(os.path.basename(f)) for f in files_to_process}
            except Exception as e:
                print(f" Error loading maps: {e}")
                return

        # --- 2. Determine Output Shape (Legacy vs RamanSPy) ---
        if self.use_ramanspy:
            # Create dummy to find new cropped length
            dummy_container = rp.SpectralContainer(np.zeros((1, len(self.wavenumbers))), self.wavenumbers)
            cropper = rp.preprocessing.misc.Cropper(region=self.CROP_REGION)
            dummy_out = cropper.apply(dummy_container)
            final_wavenumbers = dummy_out.spectral_axis
            n_features = len(final_wavenumbers)
            print(f" RamanSPy Mode: Output shape is {n_features} features (Cropped).")
        else:
            # Legacy Mode: Keep original full size
            final_wavenumbers = self.wavenumbers
            n_features = len(self.wavenumbers)
            print(f" Legacy Mode: Output shape is {n_features} features (Full).")

        # --- 3. Create H5 File ---
        with h5py.File(self.output_file, "w") as h5f:
            h5f.create_dataset("wavenumbers", data=final_wavenumbers)
            
            # Dynamic shape based on mode
            dset_spectra = h5f.create_dataset("spectra", shape=(0, n_features), 
                                              maxshape=(None, n_features), 
                                              dtype='float32', compression="gzip")
            
            # Only create 'spectra_smoothed' if in Legacy mode (RamanSPy pipeline doesn't need it)
            if not self.use_ramanspy:
                dset_spectra_smoothed = h5f.create_dataset("spectra_smoothed", shape=(0, n_features), 
                                                           maxshape=(None, n_features), 
                                                           dtype='float32', compression="gzip")

            dset_class = h5f.create_dataset("y_class", shape=(0,), maxshape=(None,), dtype='int32')
            dset_patient = h5f.create_dataset("patient_ids", shape=(0,), maxshape=(None,), dtype='int32')
            dset_samples = h5f.create_dataset("sample_ids", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
            dset_coords = h5f.create_dataset("coords", shape=(0, 2), maxshape=(None, 2), dtype='int32')

            # --- 4. Processing Loop ---
            for file_num, filepath in enumerate(tqdm(files_to_process, desc="Processing")):
                filename = os.path.basename(filepath)
                if any(ignored in filename for ignored in self.IGNORE_SAMPLES): continue
                
                match = self.filename_regex.search(filepath)
                if not match: continue
                class_id, patient_id = int(match.group("class_id")) - 1, int(match.group("patient_id"))
                sample_name = filename.replace(".npy", "")

                try:
                    raw_cube = np.load(filepath)
                    if isinstance(raw_cube, np.ndarray) and raw_cube.dtype == 'O': 
                        raw_cube = raw_cube.item().get('data', raw_cube)
                except: continue

                # A. Spatial Smoothing (Always done for mask generation)
                smoothed_cube = gaussian_filter(raw_cube, sigma=(1.0, 1.0, 0))

                # B. Mask Selection
                if self.mask_strategy == "tumor_map":
                    t_map = file_to_map.get(filepath)
                    valid_mask = (t_map == 1) if t_map is not None else None
                else:
                    valid_mask = self._create_bio_mask(smoothed_cube)
                
                if valid_mask is None or np.sum(valid_mask) == 0: continue
                
                # C. Extract Data
                y_coords, x_coords = np.where(valid_mask)
                flat_data_pure = raw_cube[valid_mask]          

                # --- D. Branching Preprocessing Logic ---
                if self.use_ramanspy:
                    # OPTION A: New Fast Way (RamanSPy)
                    # Returns cropped data automatically
                    batch_spectra_pure, _ = self._preprocess_batch_ramanspy(flat_data_pure, self.wavenumbers)
                    # We reuse pure data because RamanSPy handles smoothing internally if configured
                else:
                    # OPTION B: Old Legacy Way (Original Logic)
                    flat_data_smoothed = smoothed_cube[valid_mask]
                    
                    batch_spectra_pure = [self._preprocess_spectrum(s) for s in flat_data_pure]
                    batch_spectra_pure = np.array(batch_spectra_pure, dtype='float32')
                    
                    batch_spectra_smoothed = [self._preprocess_spectrum(s) for s in flat_data_smoothed]
                    batch_spectra_smoothed = np.array(batch_spectra_smoothed, dtype='float32')

                n_new = batch_spectra_pure.shape[0]
                if n_new == 0: continue

                # --- E. Save to H5 ---
                current_size = dset_spectra.shape[0]
                new_size = current_size + n_new
                
                # Resize datasets
                dset_spectra.resize(new_size, axis=0)
                dset_class.resize(new_size, axis=0)
                dset_patient.resize(new_size, axis=0)
                dset_samples.resize(new_size, axis=0)
                dset_coords.resize(new_size, axis=0)
                
                # Write data
                dset_spectra[current_size:] = batch_spectra_pure
                dset_class[current_size:] = np.full(n_new, class_id, dtype='int32')
                dset_patient[current_size:] = np.full(n_new, patient_id, dtype='int32')
                dset_samples[current_size:] = np.full(n_new, sample_name)
                dset_coords[current_size:] = np.column_stack((x_coords, y_coords))
                
                # Only save smoothed data if in Legacy mode
                if not self.use_ramanspy:
                    dset_spectra_smoothed.resize(new_size, axis=0)
                    dset_spectra_smoothed[current_size:] = batch_spectra_smoothed

        print(f" Processing complete. Saved to {self.output_file}")