import h5py
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

class GliomaDataLoader:
    """
    Data Loader that abstracts HDF5 reading and Cross-Validation splitting.
    
    Usage:
        loader = GliomaDataLoader(data_path="...", split_file="...")
        for fold_data in loader:
            X_train, y_train = fold_data['train']
            # ...
    """
    def __init__(self, data_path, split_file=None, memory_mapped=False):
        self.data_path = Path(data_path)
        self.split_file = Path(split_file) if split_file else None
        self.memory_mapped = memory_mapped
        
        self.h5_file = None 
        self.X = None
        self.y_class = None 
        self.groups = None # Sample IDs (e.g., "HF-448")
        
        self._load_data()

    def _load_data(self):
        print(f"📂 Loading data from {self.data_path}...")
        
        if self.memory_mapped:
            # Lazy loading (good for low RAM, slightly slower access)
            self.h5_file = h5py.File(self.data_path, 'r')
            self.X = self.h5_file['spectra']
            self.y_class = self.h5_file['y_class']
            raw_groups = self.h5_file['sample_ids'][:]
        else:
            # Load everything into RAM (fastest for training)
            with h5py.File(self.data_path, 'r') as f:
                self.X = f['spectra'][:]
                self.y_class = f['y_class'][:]
                raw_groups = f['sample_ids'][:]

        # --- FIX: Robust String Decoding ---
        # HDF5 often stores strings as bytes. We need standard Python strings
        # to compare with JSON split files.
        if isinstance(raw_groups[0], (bytes, np.bytes_)):
            self.groups = np.char.decode(raw_groups.astype('S'), 'utf-8')
        else:
            self.groups = raw_groups.astype(str)
            
        print(f"✅ Data Loaded: {self.X.shape} spectra.")

    def __iter__(self):
        """
        Iterates over cross-validation folds.
        Yields:
            dict: {
                'fold': int,
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'val_patient_ids': list
            }
        """
        # --- MODE A: Manual Splits (from JSON) ---
        if self.split_file and self.split_file.exists():
            print(f"🔄 Using Manual Splits from: {self.split_file}")
            with open(self.split_file, 'r') as f:
                splits = json.load(f)
            
            # JSON format: {'fold_0': {'train': [...], 'test': [...]}, ...}
            for i, (fold_name, split_data) in enumerate(splits.items()):
                train_samples = split_data['train']
                test_samples = split_data['test']
                
                # Create Boolean Masks based on Sample IDs
                # np.isin is optimized for this
                train_mask = np.isin(self.groups, train_samples)
                test_mask  = np.isin(self.groups, test_samples)
                
                yield {
                    'fold': i,
                    'fold_name': fold_name,
                    'train': (self.X[train_mask], self.y_class[train_mask]),
                    'val':   (self.X[test_mask],  self.y_class[test_mask]),
                    'val_sample_ids': np.unique(self.groups[test_mask])
                }
        
        # --- MODE B: Automatic Splits (Stratified Group K-Fold) ---
        else:
            print("🔄 No split file found/provided. Using StratifiedGroupKFold(n=5).")
            sgkf = StratifiedGroupKFold(n_splits=5)
            
            # We generate splits on the fly
            # Note: We pass self.groups to 'groups' argument to ensure patient independence
            for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(self.X, self.y_class, self.groups)):
                yield {
                    'fold': fold_idx,
                    'fold_name': f"fold_{fold_idx}",
                    'train': (self.X[train_idx], self.y_class[train_idx]),
                    'val':   (self.X[val_idx],   self.y_class[val_idx]),
                    'val_sample_ids': np.unique(self.groups[val_idx])
                }

    def __len__(self):
        """Returns number of folds (default 5 if unknown)."""
        if self.split_file and self.split_file.exists():
            with open(self.split_file, 'r') as f:
                return len(json.load(f))
        return 5

    def close(self):
        if self.h5_file:
            self.h5_file.close()