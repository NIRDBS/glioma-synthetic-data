import os
import glob
import re
import numpy as np
import h5py
from pathlib import Path
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import natsort

class RamanPreprocessor:
    """
    Builder class that reads raw .npy files, applies airPLS baseline correction,
    normalization, and saves the result to a single HDF5 file for fast loading.
    """
    
    # Configuration from previous experiments
    IGNORE_SAMPLES = ["HF-1887_via-t_2_1.h5_3"]
    AIRPLS_LAMBDA = 4
    AIRPLS_PORDER = 1
    
    def __init__(self, raw_data_dir, tumor_maps_path, output_file):
        """
        args:
            raw_data_dir: Path to 'data/raw' containing LGm-* folders.
            tumor_maps_path: Path to 'tumor_maps.npy'.
            output_file: Path where the processed .h5 file will be saved.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.tumor_maps_path = Path(tumor_maps_path)
        self.output_file = Path(output_file)
        
        # Compile regex for parsing filenames
        # Matches format: .../LGm-1/HF-123_desc.npy
        self.filename_regex = re.compile(r"LGm-(?P<class_id>\d)[/\\]HF-(?P<patient_id>\d+)_")

    # --- airPLS Algorithms (Copied from Notebook) ---
    def _whittaker_smooth(self, x, w, lambda_, differences=1):
        X = np.matrix(x)
        m = X.size
        E = eye(m, format='csc')
        D = E[1:] - E[:-1]
        W = diags(w, 0, shape=(m, m))
        A = csc_matrix(W + (lambda_ * D.T * D))
        B = csc_matrix(W * X.T)
        background = spsolve(A, B)
        return np.array(background)

    def _air_pls(self, x, lambda_=10, porder=1, itermax=15):
        m = x.shape[0]
        w = np.ones(m)
        for i in range(1, itermax + 1):
            z = self._whittaker_smooth(x, w, lambda_, differences=porder)
            d = x - z
            dssn = np.abs(d[d < 0].sum())
            if dssn < 0.001 * (abs(x)).sum() or i == itermax:
                break
            w[d >= 0] = 0
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
            w[0] = np.exp(i * (d[d < 0]).max() / dssn)
            w[-1] = w[0]
        return z

    def _preprocess_spectrum(self, spectrum):
        """Applies Baseline Correction + MinMax Normalization."""
        # 1. Baseline Correction
        corrected = spectrum - self._air_pls(spectrum, self.AIRPLS_LAMBDA, self.AIRPLS_PORDER)
        corrected[corrected < 0] = 0 # Clip negative values
        
        # 2. Min-Max Normalization
        min_val = np.min(corrected)
        max_val = np.max(corrected)
        if max_val - min_val == 0:
            return corrected # Avoid div by zero
        return (corrected - min_val) / (max_val - min_val)

    def run(self):
        print(f"Scanning {self.raw_data_dir}...")
        
        # 1. Find ALL files
        all_raw_files = glob.glob(str(self.raw_data_dir / "LGm*/*.npy"), recursive=True)
        
        # 2. Sort by Filename Stem (Standard Protocol)
        # This matches the order in tumor_maps.npy (0..58)
        all_raw_files = natsort.natsorted(all_raw_files, key=lambda x: Path(x).stem)
        
        # 3. Load Maps
        if self.tumor_maps_path.exists():
            print(f"Loading tumor maps from {self.tumor_maps_path}...")
            tumor_maps = np.load(self.tumor_maps_path, allow_pickle=True)
            print(f"Loaded {len(tumor_maps)} maps for {len(all_raw_files)} files.")
        else:
            print("WARNING: No tumor maps found. Setting to None.")
            tumor_maps = [None] * len(all_raw_files)

        print("Starting Streaming Process...")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # 4. Create HDF5 file
        with h5py.File(self.output_file, 'w') as f:
            # Sniff dimensions from the first file (Index 0)
            first_data = np.load(all_raw_files[0])
            n_bands = first_data.shape[-1] if first_data.ndim == 3 else first_data.shape[1]
            
            # Initialize resizable datasets
            dset_spectra = f.create_dataset('spectra', shape=(0, n_bands), maxshape=(None, n_bands), dtype='float32', chunks=True)
            dset_class = f.create_dataset('y_class', shape=(0,), maxshape=(None,), dtype='int32', chunks=True)
            dset_tumor = f.create_dataset('y_tumor', shape=(0,), maxshape=(None,), dtype='int32', chunks=True)
            dset_patient = f.create_dataset('patient_ids', shape=(0,), maxshape=(None,), dtype='int32', chunks=True)
            
            dt_str = h5py.string_dtype(encoding='utf-8')
            dset_samples = f.create_dataset('sample_ids', shape=(0,), maxshape=(None,), dtype=dt_str, chunks=True)

            # 5. Iterate and Write
            # CRITICAL: We zip() the full sorted lists.
            # This ensures File[24] pairs with Map[24].
            # The loop stops when tumor_maps is exhausted (ignoring the 60th file which has no map).
            
            pairs = list(zip(all_raw_files, tumor_maps))
            
            for file_path, t_map in tqdm(pairs):
                path_obj = Path(file_path)
                sample_name = path_obj.stem
                
                # CHECK 1: Is this the ignored sample?
                # We identify it, but we continue the loop to keep the zip alignment valid.
                if sample_name in self.IGNORE_SAMPLES:
                    # print(f"Ignoring sample: {sample_name}")
                    continue
                    
                # Metadata Parsing
                match = self.filename_regex.search(str(path_obj))
                if not match: 
                    print(f"Skipping {sample_name}: Regex match failed.")
                    continue
                class_id = int(match.group("class_id"))
                patient_id = int(match.group("patient_id"))

                # Load Data
                raw_data = np.load(file_path)
                
                # Flatten Logic
                if raw_data.ndim == 3:
                    n_pixels = raw_data.shape[0] * raw_data.shape[1]
                    flat_data = raw_data.reshape(n_pixels, -1)
                    
                    try:
                        flat_t_map = t_map.reshape(n_pixels) if t_map is not None else np.zeros(n_pixels)
                    except ValueError:
                        print(f"Error aligning {sample_name}: Map size {t_map.size} != Data size {n_pixels}")
                        # If a single file is malformed, we skip it but don't crash the whole run
                        continue
                else:
                    flat_data = raw_data
                    try:
                        flat_t_map = t_map.flatten() if t_map is not None else np.zeros(flat_data.shape[0])
                    except ValueError:
                         print(f"Error aligning {sample_name}: Map size {t_map.size} != Data size {flat_data.shape[0]}")
                         continue

                # Preprocess Batch
                batch_spectra = []
                for i in range(flat_data.shape[0]):
                    batch_spectra.append(self._preprocess_spectrum(flat_data[i]))
                
                batch_spectra = np.array(batch_spectra, dtype='float32')
                n_new = batch_spectra.shape[0]

                # Append to HDF5
                current_size = dset_spectra.shape[0]
                new_size = current_size + n_new
                
                dset_spectra.resize(new_size, axis=0)
                dset_class.resize(new_size, axis=0)
                dset_tumor.resize(new_size, axis=0)
                dset_patient.resize(new_size, axis=0)
                dset_samples.resize(new_size, axis=0)
                
                dset_spectra[current_size:] = batch_spectra
                dset_class[current_size:] = np.full(n_new, class_id, dtype='int32')
                dset_tumor[current_size:] = flat_t_map.astype('int32')
                dset_patient[current_size:] = np.full(n_new, patient_id, dtype='int32')
                dset_samples[current_size:] = np.full(n_new, sample_name, dtype=object)

        print("Done.")