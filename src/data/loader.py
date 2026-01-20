import h5py
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

class GliomaDataLoader:
    def __init__(self, data_path, split_file=None, memory_mapped=False):
        self.data_path = data_path
        self.split_file = split_file
        self.memory_mapped = memory_mapped
        self.h5_file = None 
        
        self.X = None
        self.y_class = None 
        self.y_tumor = None 
        self.groups = None
        
        self._load_data()

    def _load_data(self):
        print(f"Loading data from {self.data_path}...")
        
        if self.memory_mapped:
            self.h5_file = h5py.File(self.data_path, 'r')
            self.X = self.h5_file['spectra']
            self.y_class = self.h5_file['y_class']
            self.y_tumor = self.h5_file['y_tumor']
            raw_groups = self.h5_file['sample_ids'][:]
        else:
            with h5py.File(self.data_path, 'r') as f:
                self.X = f['spectra'][:]
                self.y_class = f['y_class'][:]
                self.y_tumor = f['y_tumor'][:]
                raw_groups = f['sample_ids'][:]

        # --- FIX: Robust String Decoding ---
        # 1. If it's already object/unicode, ensure it's a numpy array of str
        # 2. If it's bytes (S), decode it to utf-8
        if raw_groups.dtype.kind == 'S':
            self.groups = np.char.decode(raw_groups, 'utf-8')
        elif raw_groups.dtype.kind == 'O':
            # Handle mixed bytes/strings in object array
            self.groups = np.array([
                s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                for s in raw_groups
            ])
        else:
            self.groups = raw_groups.astype(str)

        print(f"Data Loaded. Shape: {self.X.shape}")
        print(f"Unique Samples: {len(np.unique(self.groups))}")
        print(f"Sample ID Example: '{self.groups[0]}' (Type: {type(self.groups[0])})")

    def close(self):
        if self.h5_file:
            self.h5_file.close()

    def get_cross_validation_generator(self, n_splits=5):
        if self.split_file:
            print(f"Loading frozen splits from {self.split_file}")
            with open(self.split_file, 'r') as f:
                splits = json.load(f)
                
            for fold_name, split_data in splits.items():
                train_samples = split_data['train'] # List of strings
                test_samples = split_data['test']   # List of strings
                
                # Debugging print to ensure matching works
                if fold_name == 'fold_0':
                     print(f"DEBUG: Split asks for '{train_samples[0]}'")
                     print(f"DEBUG: Loader has '{self.groups[0]}'")
                
                train_mask = np.isin(self.groups, train_samples)
                test_mask = np.isin(self.groups, test_samples)
                
                yield {
                    'fold': fold_name,
                    'train': (self.X[train_mask], self.y_class[train_mask], self.y_tumor[train_mask]),
                    'val':   (self.X[test_mask],  self.y_class[test_mask],  self.y_tumor[test_mask]),
                    'val_sample_ids': self.groups[test_mask]
                }
        else:
            # Fallback to random split
            sgkf = StratifiedGroupKFold(n_splits=n_splits)
            for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(self.X, self.y_class, self.groups)):
                yield {
                    'fold': fold_idx,
                    'train': (self.X[train_idx], self.y_class[train_idx], self.y_tumor[train_idx]),
                    'val': (self.X[val_idx], self.y_class[val_idx], self.y_tumor[val_idx]),
                    'val_sample_ids': self.groups[val_idx] 
                }