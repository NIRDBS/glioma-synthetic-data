import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

class GliomaDataLoader:
    """
    A robust data manager for Raman Spectra experiments.
    """

    def __init__(self, data_path, split_file=None, memory_mapped=False):
        """
        args:
            data_path: Path to the processed .h5 file.
            memory_mapped: If True, keeps data on disk. If False, loads to RAM.
            split_file: Path to a JSON file containing manual splits (optional).
        """
        self.data_path = data_path
        self.split_file = split_file # Path to json
        self.memory_mapped = memory_mapped
        
        self.X = None
        self.y_class = None 
        self.y_tumor = None 
        self.groups = None
        
        self._load_data()

    def _load_data(self):
        """Internal method to load data from the HDF5 artifact."""
        print(f"Loading data from {self.data_path}...")
        
        with h5py.File(self.data_path, 'r') as f:
            # FIX: Keys must match what RamanPreprocessor saved
            if not self.memory_mapped:
                self.X = f['spectra'][:]
                self.y_class = f['y_class'][:]   # Was 'labels_class'
                self.y_tumor = f['y_tumor'][:]   # Was 'labels_tumor'
                
                # Decode sample IDs from bytes to strings if necessary
                raw_groups = f['sample_ids'][:]
                if raw_groups.dtype.kind == 'S': # Check if byte string
                    self.groups = raw_groups.astype(str)
                else:
                    self.groups = raw_groups
            else:
                # Keep on disk (advanced usage)
                self.X = f['spectra']
                self.y_class = f['y_class']
                self.y_tumor = f['y_tumor']
                self.groups = f['sample_ids']

        print(f"Data Loaded. Shape: {self.X.shape}")
        
        # Check if groups are loaded
        if self.groups is not None:
             print(f"Unique Samples (Groups): {len(np.unique(self.groups))}")
        else:
            print("Warning: Sample IDs (Groups) not loaded correctly.")

    def get_cross_validation_generator(self, n_splits=5):
        """
        Yields train/test splits.
        If self.split_file is set, it reads from the JSON.
        If not, it generates a fresh StratifiedGroupKFold.
        """
        
        # OPTION A: Load Frozen Split
        if self.split_file:
            print(f"Loading frozen splits from {self.split_file}")
            with open(self.split_file, 'r') as f:
                splits = json.load(f)
                
            for fold_name, split_data in splits.items():
                train_samples = set(split_data['train'])
                test_samples = set(split_data['test'])
                
                # Convert string IDs (in self.groups) to boolean masks
                # This is fast enough for 300k items
                train_mask = np.isin(self.groups, list(train_samples))
                test_mask = np.isin(self.groups, list(test_samples))
                
                yield {
                    'fold': fold_name,
                    'train': (self.X[train_mask], self.y_class[train_mask], self.y_tumor[train_mask]),
                    'val':   (self.X[test_mask],  self.y_class[test_mask],  self.y_tumor[test_mask]),
                    'val_sample_ids': self.groups[test_mask]
                }
                
        # OPTION B: Generate Fresh Split (Fallback)
        else:
            print("No split file provided. Generating fresh StratifiedGroupKFold...")
            sgkf = StratifiedGroupKFold(n_splits=n_splits)
            for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(self.X, self.y_class, self.groups)):
                yield {
                    'fold': fold_idx,
                    'train': (self.X[train_idx], self.y_class[train_idx], self.y_tumor[train_idx]),
                    'val': (self.X[val_idx], self.y_class[val_idx], self.y_tumor[val_idx]),
                    'val_sample_ids': self.groups[val_idx] 
                }

    def make_tf_dataset(self, x_data, y_cond_class, y_cond_tumor, batch_size=32, shuffle=True):
        """
        Helper to convert numpy arrays to TensorFlow Datasets for CVAE training.
        """
        # Ensure types are correct for TF
        x_data = x_data.astype(np.float32)
        y_cond_class = y_cond_class.astype(np.int32)
        y_cond_tumor = y_cond_tumor.astype(np.int32)

        # Expand dims for Conv1D input: (N, 1000) -> (N, 1000, 1)
        if x_data.ndim == 2:
            x_data = np.expand_dims(x_data, axis=-1)
            
        dataset = tf.data.Dataset.from_tensor_slices((x_data, (y_cond_class, y_cond_tumor)))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Usage Example (Mental Model) ---
# manager = GliomaDataLoader("data/processed/dataset_v1.h5")
#
# for fold_data in manager.get_cross_validation_generator():
#     X_train, y_class_train, _ = fold_data['train']
#     
#     # Train your SVM
#     svm.fit(X_train, y_class_train)