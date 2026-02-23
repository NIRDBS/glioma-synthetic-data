import h5py
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

class GliomaDataLoader:
    """
    Data Loader that abstracts HDF5 reading and Cross-Validation splitting.
    Now supports filtering out healthy tissue using 'tumor_only=True'.
    """
    def __init__(self, data_path, split_file=None, memory_mapped=False, tumor_only=False):
        self.data_path = Path(data_path)
        self.split_file = Path(split_file) if split_file else None
        self.memory_mapped = memory_mapped
        self.tumor_only = tumor_only # <--- NEW FLAG
        
        self.h5_file = None 
        self.X = None
        self.y_class = None 
        self.y_tumor = None
        self.wavenumbers = None # <--- Preserved for your plotting code
        self.groups = None 
        
        self._load_data()

    def _load_data(self):
        print(f"📂 Loading data from {self.data_path}...")
        
        if self.memory_mapped:
            self.h5_file = h5py.File(self.data_path, 'r')
            self.X = self.h5_file['spectra']
            self.y_class = self.h5_file['y_class']
            if 'y_tumor' in self.h5_file:
                self.y_tumor = self.h5_file['y_tumor']
            if 'wavenumbers' in self.h5_file:
                self.wavenumbers = self.h5_file['wavenumbers'][:]
            raw_groups = self.h5_file['sample_ids'][:]
        else:
            with h5py.File(self.data_path, 'r') as f:
                self.X = f['spectra'][:]
                self.y_class = f['y_class'][:]
                if 'y_tumor' in f:
                    self.y_tumor = f['y_tumor'][:]
                if 'wavenumbers' in f:
                    self.wavenumbers = f['wavenumbers'][:]
                raw_groups = f['sample_ids'][:]

        # Robust String Decoding
        if isinstance(raw_groups[0], (bytes, np.bytes_)):
            self.groups = np.char.decode(raw_groups.astype('S'), 'utf-8')
        else:
            self.groups = raw_groups.astype(str)
            
        # --- TUMOR FILTERING LOGIC ---
        if self.tumor_only:
            if self.y_tumor is not None:
                print("🔬 'tumor_only=True': Filtering out healthy tissue...")
                # Convert to numpy array in case it's an HDF5 dataset ref
                y_t = self.y_tumor[:] 
                mask = (y_t == 0)
                
                # Apply mask to all arrays
                self.X = self.X[mask]
                self.y_class = self.y_class[mask]
                self.y_tumor = self.y_tumor[mask]
                self.groups = self.groups[mask]
                print(f"   ✅ Filtered to {self.X.shape[0]} Pure Tumor Spectra.")
            else:
                print("   ⚠️ WARNING: 'y_tumor' not found in HDF5! Using all data.")
        else:
            print(f"✅ Loaded {self.X.shape[0]} Total Spectra (Tumor + Healthy).")

    def __iter__(self):
        if self.split_file and self.split_file.exists():
            print(f"🔄 Using Manual Splits from: {self.split_file}")
            with open(self.split_file, 'r') as f:
                splits = json.load(f)
            
            for i, (fold_name, split_data) in enumerate(splits.items()):
                train_samples = split_data['train']
                test_samples = split_data['test']
                
                train_mask = np.isin(self.groups, train_samples)
                test_mask  = np.isin(self.groups, test_samples)
                
                yield {
                    'fold': i,
                    'fold_name': fold_name,
                    'train': (self.X[train_mask], self.y_class[train_mask]),
                    'val':   (self.X[test_mask],  self.y_class[test_mask]),
                    'val_sample_ids': np.unique(self.groups[test_mask])
                }
        else:
            print("🔄 Using StratifiedGroupKFold(n=5).")
            sgkf = StratifiedGroupKFold(n_splits=5)
            for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(self.X, self.y_class, self.groups)):
                yield {
                    'fold': fold_idx,
                    'fold_name': f"fold_{fold_idx}",
                    'train': (self.X[train_idx], self.y_class[train_idx]),
                    'val':   (self.X[val_idx],   self.y_class[val_idx]),
                    'val_sample_ids': np.unique(self.groups[val_idx])
                }

    def __len__(self):
        if self.split_file and self.split_file.exists():
            with open(self.split_file, 'r') as f: return len(json.load(f))
        return 5

    def close(self):
        if self.h5_file: self.h5_file.close()