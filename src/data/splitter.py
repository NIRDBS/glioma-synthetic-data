import json
import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

class Splitter:
    """
    Responsible for defining Train/Val/Test sets and saving them to disk.
    """
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_existing_groups(self):
        """Reads the H5 file to find all unique sample IDs and their classes."""
        with h5py.File(self.data_path, 'r') as f:
            # We need to map Sample ID -> Class (for stratification)
            # Since all spectra in a sample have the same class, we just grab the first one.
            
            # This is a bit slow if we iterate 300k items. 
            # Optimization: Iterate unique samples if possible, or just scan once.
            # Assuming 'sample_ids' and 'y_class' are aligned.
            
            sample_ids = f['sample_ids'][:].astype(str)
            y_class = f['y_class'][:]
            
            # Create a dictionary: { 'HF-448': class_1, 'HF-901': class_2 }
            # We assume a sample doesn't have mixed classes.
            unique_samples, indices = np.unique(sample_ids, return_index=True)
            unique_classes = y_class[indices]
            
            return unique_samples, unique_classes

    def generate_manual_split(self):
        """
        Saves the specific 5-fold split defined in the previous project.
        """
        # COPY-PASTED FROM YOUR NOTEBOOK CONTENT
        CV5folds = [
            # Fold 1
            ["HF-448_V5B_1.h5_3", "HF-305_v4b_1_1.h5_6", "HF-615_V5BB_1.h5_9", "HF-2104_#5_1.h5_0", 
             "HF-2104_#9_1.h5_1", "HF-2104_V1T_1.h5_2", "HF-442_V4BB_1.h5_12", "HF-1002_V1AT_1.h5_0", 
             "HF-1002_V2AT_1.h5_1", "HF-682_V3AT_1.h5_9", "HF-682_V3BB_1.h5_10", "HF-894_9_1.h5_11", 
             "HF-894_V1BB_1.h5_12", "HF-592_V3T_1.h5_4"],
            # Fold 2
            ["HF-868_1_2.h5_4", "HF-901_V2T_2.h5_10", "HF-960_VIAT_2.h5_11", "HF-2614_V1B_1.h5_3", 
             "HF-1825_V2B_1.h5_2", "HF-2102_V2BB_1.h5_3", "HF-2102_V3AM_1.h5_4", "HF-2102_V3AM_2.h5_5", 
             "HF-988_V1-T_1.h5_13", "HF-988_V1B_1.h5_14", "HF-1043_V1AM_1.h5_0", "HF-2106_V3AM_1.h5_0"],
            # Fold 3
            ["HF-1293_13_1.h5_0", "HF-1010_V1T_1.h5_0", "HF-1016_IAT_2.h5_1", "HF-1334_V58-B_2_1.h5_2", 
             "HF-2849_VIT2_1.h5_4", "HF-2849_VIT2_1.h5_5", "HF-2849_VIT2_2.h5_6", "HF-2849_VIT_2_new2021.h5_7", 
             "HF-2454_V1AT_1.h5_6", "HF-2548_V1T_1.h5_7", "HF-1086_#1_1.h5_1", "HF-2355_V2AM_1.h5_2", 
             "HF-2493_V1T_1.h5_1", "HF-2493_V1T_2.h5_2"],
            # Fold 4
            ["HF-1295_V3AM_2.h5_1", "HF-2070_V1T_1.h5_4", "HF-2776_V2B_2.h5_5", "HF-2852_VIT_2_2.h5_8", 
             "HF-2715_VIL_1.h5_8", "HF-2802_V3T_1.h5_9", "HF-2485_V1B_1.h5_3", "HF-2600_V1B_1.h5_4", 
             "HF-2608_V1T_1.h5_5", "HF-2544_V1B_1.h5_3"],
            # Fold 5
            ["HF-2534_V2B_1.h5_2", "HF-3271_VIB_2.h5_7", "HF-3337_V3T_1.h5_8", "HF-2104_#5_1.h5_0", 
             "HF-2104_#9_1.h5_1", "HF-2104_V1T_1.h5_2", "HF-2876_V1T_1.h5_10", "HF-2898_V1T_1.h5_11", 
             "HF-2619_V1T_1.h5_6", "HF-2619_V4T_1.h5_7", "HF-2666_V2B_1.h5_8", "HF-592_V3T_1.h5_4"]
        ]

        splits_data = {}
        
        # In this manual split, CV5folds contains the TEST samples for that fold.
        # The TRAIN samples are everyone else.
        
        # 1. Get all available samples from the H5 file to know the "Universe" of samples
        all_samples, _ = self.get_existing_groups()
        all_samples_set = set(all_samples)
        
        for fold_idx, test_samples in enumerate(CV5folds):
            # Clean up names if necessary (strip whitespace)
            test_samples = [s.strip() for s in test_samples]
            
            # Calculate Train = Universe - Test
            train_samples = list(all_samples_set - set(test_samples))
            
            splits_data[f"fold_{fold_idx}"] = {
                "train": train_samples,
                "test": test_samples
            }
            
        save_path = self.output_dir / "manual_splits.json"
        with open(save_path, 'w') as f:
            json.dump(splits_data, f, indent=4)
            
        print(f"Saved manual splits to {save_path}")

    def generate_random_split(self, n_splits=5, seed=42):
        """Generates a reproducible random Stratified Group K-Fold split."""
        unique_samples, unique_classes = self.get_existing_groups()
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        splits_data = {}
        
        # Mock X (zeros) just to satisfy the sklearn API
        mock_X = np.zeros(len(unique_samples))
        
        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(mock_X, unique_classes, unique_samples)):
            splits_data[f"fold_{fold_idx}"] = {
                "train": unique_samples[train_idx].tolist(),
                "test": unique_samples[test_idx].tolist()
            }
            
        save_path = self.output_dir / "random_splits.json"
        with open(save_path, 'w') as f:
            json.dump(splits_data, f, indent=4)
        
        print(f"Saved random splits to {save_path}")