import os
import glob
import re
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import natsort

class RamanPreprocessor:
    """
    Final Preprocessor.
    strategies:
      - 'bio_mask': Physics-based (Contrast + Density). Robust and preferred.
      - 'tumor_map': Uses Ground Truth maps, matched by Image Dimensions (Fallback).
    """
    
    IGNORE_SAMPLES = ["HF-1887_via-t_2_1.h5_3"]
    AIRPLS_LAMBDA = 10000  # High stiffness to preserve biological peaks
    AIRPLS_PORDER = 1
    SIGNAL_LENGTH = 1738 

    def __init__(self, raw_data_dir, output_file, 
                 mask_strategy="bio_mask",
                 strict_mask=False, # <--- NEW TOGGLE
                 tumor_maps_path=None, 
                 calibration_path=None):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.output_file = Path(output_file)
        self.mask_strategy = mask_strategy
        self.strict_mask = strict_mask
        self.tumor_maps_path = Path(tumor_maps_path) if tumor_maps_path else None
        self.calibration_path = Path(calibration_path) if calibration_path else None

        self.filename_regex = re.compile(r"LGm-(?P<class_id>\d)[/\\]HF-(?P<patient_id>\d+)_")
        
        if self.calibration_path:
            self._load_calibration()
            
        self.m = self.SIGNAL_LENGTH
        self.D = np.diff(np.eye(self.m), 2)
        self.DD = self.D.dot(self.D.T)
        self.DD_sparse = csc_matrix(self.DD)

    def _load_calibration(self):
        try:
            df = pd.read_csv(self.calibration_path)
            if 'freq' in df.columns: wavenumbers = df['freq'].values
            else: wavenumbers = df.iloc[:, 1].values
            
            if len(wavenumbers) < self.SIGNAL_LENGTH:
                 diff = np.mean(np.diff(wavenumbers[-10:]))
                 padding = [wavenumbers[-1] + diff * i for i in range(1, self.SIGNAL_LENGTH - len(wavenumbers) + 1)]
                 wavenumbers = np.concatenate([wavenumbers, padding])
            self.wavenumbers = wavenumbers
            
            self.idx_lipid_start = np.argmin(np.abs(wavenumbers - 2850))
            self.idx_lipid_end   = np.argmin(np.abs(wavenumbers - 2950))
            self.idx_silent_start = np.argmin(np.abs(wavenumbers - 2100))
            self.idx_silent_end   = np.argmin(np.abs(wavenumbers - 2200))
            
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            raise

    def _create_bio_mask(self, image_data):
        """
        BIO-MASK (Physics + Density).
        If strict_mask=True, applies a median filter to keep only dense tissue.
        If strict_mask=False, keeps all biological tissue.
        """
        lipid_map = np.mean(image_data[:, :, self.idx_lipid_start:self.idx_lipid_end], axis=2)
        silent_map = np.mean(image_data[:, :, self.idx_silent_start:self.idx_silent_end], axis=2)
        contrast_map = lipid_map - silent_map
        
        basic_mask = contrast_map > 5.0
        
        if self.strict_mask:
            if np.sum(basic_mask) > 0:
                median_val = np.median(contrast_map[basic_mask])
                strict_threshold = max(5.0, median_val)
                final_mask = contrast_map > strict_threshold
            else:
                final_mask = basic_mask
        else:
            final_mask = basic_mask

        return binary_erosion(final_mask, iterations=1)

    def _build_file_to_map_correspondence(self, files_to_process, tumor_maps_array):
        file_to_map = {}
        used_map_indices = set()
        print("🔗 Matching files to tumor maps by SHAPE...")
        
        for filepath in files_to_process:
            try:
                raw_cube = np.load(filepath)
                if isinstance(raw_cube, np.ndarray) and raw_cube.dtype == 'O':
                    raw_cube = raw_cube.item().get('data', raw_cube)
                cube_h, cube_w = raw_cube.shape[:2]
                
                matched_map, matched_idx = None, None
                
                for map_idx, t_map in enumerate(tumor_maps_array):
                    if map_idx in used_map_indices: continue
                    if t_map.shape[:2] == (cube_h, cube_w):
                        matched_map = t_map
                        matched_idx = map_idx
                        break
                
                if matched_map is not None:
                    file_to_map[filepath] = matched_map
                    used_map_indices.add(matched_idx)
                else:
                    file_to_map[filepath] = None
            except: 
                file_to_map[filepath] = None
        
        print(f"✅ Matched {len(used_map_indices)} / {len(files_to_process)} files to maps.")
        return file_to_map

    def _preprocess_spectrum(self, raw_spectrum):
        baseline = self._airPLS(raw_spectrum, lambda_=self.AIRPLS_LAMBDA, porder=self.AIRPLS_PORDER)
        corrected = raw_spectrum - baseline
        corrected = np.maximum(corrected, 0)
        min_val, max_val = np.min(corrected), np.max(corrected)
        if max_val - min_val == 0: return np.zeros_like(corrected)
        return (corrected - min_val) / (max_val - min_val)

    def _airPLS(self, x, lambda_, porder=1, itermax=15):
        m = x.shape[0]
        w = np.ones(m)
        for i in range(1, itermax + 1):
            W = diags(w)
            LHS = W + (lambda_ * self.DD_sparse)
            z = spsolve(LHS, w * x)
            d = x - z
            dssn = np.abs(d[d < 0].sum())
            if dssn < 0.001 * (np.abs(x).sum()): break
            w[d >= 0] = 0
            neg_idx = d < 0
            if neg_idx.any():
                w[neg_idx] = np.exp(i * np.abs(d[neg_idx]) / dssn)
            w[0] = np.exp(i * (d[d < 0]).max() / dssn) if (d<0).any() else 0
            w[-1] = w[0]
        return z

    def run(self):
        files_to_process = []
        for class_dir in glob.glob(str(self.raw_data_dir / "LGm-*")):
            files_to_process.extend(glob.glob(os.path.join(class_dir, "*.npy")))
        files_to_process = natsort.natsorted(files_to_process)
        print(f"🔍 Found {len(files_to_process)} scan files.")
        
        file_to_map = {}
        if self.mask_strategy == "tumor_map":
            print(f"📂 Loading Tumor Maps from {self.tumor_maps_path}...")
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
                print(f"❌ Error loading maps: {e}")
                return

        with h5py.File(self.output_file, "w") as h5f:
            h5f.create_dataset("wavenumbers", data=self.wavenumbers)
            
            dset_spectra = h5f.create_dataset("spectra", shape=(0, 1738), maxshape=(None, 1738), dtype='float32', compression="gzip")
            dset_class = h5f.create_dataset("y_class", shape=(0,), maxshape=(None,), dtype='int32')
            dset_patient = h5f.create_dataset("patient_ids", shape=(0,), maxshape=(None,), dtype='int32')
            dset_samples = h5f.create_dataset("sample_ids", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
            dset_coords = h5f.create_dataset("coords", shape=(0, 2), maxshape=(None, 2), dtype='int32')

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

                # --- 1. SPATIAL SMOOTHING ---
                smoothed_cube = gaussian_filter(raw_cube, sigma=(1.0, 1.0, 0))

                # --- 2. MASK SELECTION ---
                if self.mask_strategy == "tumor_map":
                    t_map = file_to_map.get(filepath)
                    valid_mask = (t_map == 1) if t_map is not None else None
                else:
                    valid_mask = self._create_bio_mask(smoothed_cube)
                
                if valid_mask is None or np.sum(valid_mask) == 0: continue
                
                # Extract coordinates and data
                y_coords, x_coords = np.where(valid_mask)
                flat_data = smoothed_cube[valid_mask]
                
                batch_spectra = [self._preprocess_spectrum(s) for s in flat_data]
                batch_spectra = np.array(batch_spectra, dtype='float32')
                n_new = batch_spectra.shape[0]

                if n_new == 0: continue

                current_size = dset_spectra.shape[0]
                new_size = current_size + n_new
                
                dset_spectra.resize(new_size, axis=0)
                dset_class.resize(new_size, axis=0)
                dset_patient.resize(new_size, axis=0)
                dset_samples.resize(new_size, axis=0)
                dset_coords.resize(new_size, axis=0)
                
                dset_spectra[current_size:] = batch_spectra
                dset_class[current_size:] = np.full(n_new, class_id, dtype='int32')
                dset_patient[current_size:] = np.full(n_new, patient_id, dtype='int32')
                dset_samples[current_size:] = np.full(n_new, sample_name)
                dset_coords[current_size:] = np.column_stack((x_coords, y_coords))

        print(f"✅ Processing complete. Saved to {self.output_file}")