import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from typing import Optional, Dict, Any, Tuple


class GliomaDataLoader:
    """
    Data Loader that abstracts HDF5 reading and Cross-Validation splitting.
    
    Features:
    - Supports group-aware CV splits from JSON files
    - Filters out healthy tissue if needed (tumor_only=True)
    - Supports config-driven split loading via load_from_config()
    - Logs which splits are being used for reproducibility
    """
    def __init__(self, data_path, split_file=None, memory_mapped=False, tumor_only=False,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data loader.

        Args:
            data_path: Path to .h5 dataset file
            split_file: Optional path to JSON file with CV splits
            memory_mapped: If True, load data lazily via HDF5; if False, load into RAM
            tumor_only: If True, filter out healthy tissue (requires y_tumor in H5)
            config: Optional configuration dict with 'cv_strategy' section
        """
        self.data_path = Path(data_path)
        self.split_file = Path(split_file) if split_file else None
        self.memory_mapped = memory_mapped
        self.tumor_only = tumor_only
        self.config = config
        
        # If config provided, extract split_source
        if config and 'cv_strategy' in config:
            cv_config = config['cv_strategy']
            if 'split_source' in cv_config:
                # split_source is relative to project root, resolve it
                self._resolve_split_path(cv_config['split_source'])
        
        self.h5_file = None
        self.X = None
        self.y_class = None
        self.y_tumor = None
        self.wavenumbers = None
        self.groups = None
        self.coords = None

        self._load_data()
    
    def _resolve_split_path(self, split_source: str) -> None:
        """
        Resolve relative split path to absolute path.
        
        Args:
            split_source: Relative path like 'data/splits/strict_splits.json'
        """
        split_source_path = Path(split_source)
        
        # If already absolute, use it
        if split_source_path.is_absolute():
            self.split_file = split_source_path
            return
        
        # Otherwise, resolve relative to project root
        # Look for project root by finding configs/ directory
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / 'configs').exists():
                self.split_file = parent / split_source_path
                return
        
        # Fallback to cwd
        self.split_file = current / split_source_path

    @classmethod
    def from_config(cls, h5_data_path: str, config: Dict[str, Any], 
                    memory_mapped: bool = False) -> 'GliomaDataLoader':
        """
        Create a GliomaDataLoader from a configuration dictionary.
        
        This is the recommended way to instantiate loaders when using
        the centralized config system.
        
        Features:
        - Applies spectral cropping if preprocessing.crop_region is specified
        - Applies patient cap downsampling if preprocessing.patient_cap is specified
        
        Args:
            h5_data_path: Path to the .h5 dataset file
            config: Configuration dict (from ConfigManager)
            memory_mapped: Whether to use lazy loading
        
        Returns:
            Initialized GliomaDataLoader instance with preprocessing applied
        
        Raises:
            ValueError: If config is missing required fields or cv_strategy.split_source
        
        Note:
            The 'y_tumor' setting is read from config['preprocessing']['y_tumor'].
            - y_tumor: true → filter to tumor-only spectra
            - y_tumor: false → use all spectra (no filtering)
        
        Example:
            from src.utils.config_manager import ConfigManager
            config = ConfigManager.load_config('exp_608')
            loader = GliomaDataLoader.from_config('data/processed/glioma_dataset.h5', config)
            # Loader will automatically apply crop_region and patient_cap if in config
        """
        if 'cv_strategy' not in config:
            raise ValueError("Config must contain 'cv_strategy' section")
        
        cv_config = config['cv_strategy']
        if 'split_source' not in cv_config:
            raise ValueError("cv_strategy must contain 'split_source' (path to splits JSON)")
        
        split_source = cv_config['split_source']
        print(f"Loading CV splits from: {split_source}")
        
        # Extract tumor_only flag from preprocessing config (defaults to True for backward compatibility)
        use_tumor_only = config.get('preprocessing', {}).get('y_tumor', True)

        # Create loader with config
        loader = cls(
            data_path=h5_data_path,
            split_file=None,  # Will be set by _resolve_split_path
            memory_mapped=memory_mapped,
            tumor_only=use_tumor_only,
            config=config,
        )
        
        # Apply exp_103+ features if specified in config
        preproc_cfg = config.get('preprocessing', {})
        
        # Apply spectral cropping first
        if 'crop_region' in preproc_cfg:
            calib_file = config.get('paths', {}).get('calib_file')
            loader.apply_spectral_crop(crop_region=preproc_cfg['crop_region'], 
                                      calib_file=calib_file)
        
        # NOTE: patient cap can be applied here, at loading, or
        # it can be applied dynamically at epoch level (see augmentations.py) for more flexibility., 
        cv_cfg = config.get('cv_strategy', {})
        if 'patient_cap' in cv_cfg and 'cap_strategy' in cv_cfg and cv_cfg['cap_strategy'] == "fixed":
            loader.apply_patient_cap(patient_cap=cv_cfg['patient_cap'])
        
        return loader
    
    def _load_data(self):
        """Load data from HDF5 file and apply tumor filtering if needed."""
        print(f"Loading data from {self.data_path}...")
        
        # 'y_bag' is the bag/slide label broadcast to its instances (never an
        # instance-level label). Older exports only have 'y_class'; read that as a
        # fallback so this loader still opens them.
        if self.memory_mapped:
            # Lazy loading (good for low RAM, slightly slower access)
            self.h5_file = h5py.File(self.data_path, 'r')
            self.X = self.h5_file['spectra']
            self.y_class = self.h5_file['y_bag'] if 'y_bag' in self.h5_file else self.h5_file['y_class']
            if 'y_tumor' in self.h5_file:
                self.y_tumor = self.h5_file['y_tumor']
            if 'wavenumbers' in self.h5_file:
                self.wavenumbers = self.h5_file['wavenumbers'][:]
            if 'coords' in self.h5_file:
                self.coords = self.h5_file['coords'][:]
            raw_groups = self.h5_file['sample_ids'][:]
        else:
            # Load everything into RAM (fastest for training)
            with h5py.File(self.data_path, 'r') as f:
                self.X = f['spectra'][:]
                self.y_class = f['y_bag'][:] if 'y_bag' in f else f['y_class'][:]
                if 'y_tumor' in f:
                    self.y_tumor = f['y_tumor'][:]
                if 'wavenumbers' in f:
                    self.wavenumbers = f['wavenumbers'][:]
                if 'coords' in f:
                    self.coords = f['coords'][:]
                raw_groups = f['sample_ids'][:]

        # Robust String Decoding
        if isinstance(raw_groups[0], (bytes, np.bytes_)):
            self.groups = np.char.decode(raw_groups.astype('S'), 'utf-8')
        else:
            self.groups = raw_groups.astype(str)
            
        # --- TUMOR FILTERING LOGIC ---
        if self.tumor_only:
            if self.y_tumor is not None:
                print("[*] 'tumor_only=True': Filtering out healthy tissue...")
                # Convert to numpy array in case it's an HDF5 dataset ref
                y_t = self.y_tumor[:] 
                mask = (y_t == 1)
                
                # Apply mask to all arrays
                self.X = self.X[mask]
                self.y_class = self.y_class[mask]
                self.y_tumor = self.y_tumor[mask]
                self.groups = self.groups[mask]
                if self.coords is not None:
                    self.coords = self.coords[mask]
                print(f"    Filtered to {self.X.shape[0]} Pure Tumor Spectra.")
            else:
                print("   ! WARNING: 'y_tumor' not found in HDF5! Using all data.")
        else:
            print(f" Loaded {self.X.shape[0]} Total Spectra (Tumor + Healthy).")

    def get_wavenumber_indices(self, crop_region: Optional[Tuple[float, float]] = None,
                              calib_file: Optional[str] = None) -> Tuple[int, int]:
        """
        Convert Raman shifts (wavenumbers) to array indices.
        
        For exp_103+: Scientists specify crop_region=[700, 1800] in cm⁻¹.
        This function finds the closest array indices to those wavenumbers
        using the RamanFreqTable.csv calibration file.
        
        Args:
            crop_region: Tuple of (start_cm1, end_cm1) e.g., (700, 1800)
            calib_file: Path to RamanFreqTable.csv with idx,freq columns
        
        Returns:
            Tuple of (start_index, end_index) for slicing X[:, start_idx:end_idx]
        
        Example:
            start_idx, end_idx = loader.get_wavenumber_indices((700, 1800), 'data/RamanFreqTable.csv')
            X_cropped = loader.X[:, start_idx:end_idx]
        """
        if crop_region is None:
            # No cropping requested
            return 0, self.X.shape[1]
        
        if calib_file is None:
            # Try to find from config
            if self.config and 'paths' in self.config and 'calib_file' in self.config['paths']:
                calib_file = self.config['paths']['calib_file']
            else:
                print("  Warning: crop_region specified but no calib_file found. Using full spectrum.")
                return 0, self.X.shape[1]
        
        # Resolve calib_file path
        calib_path = Path(calib_file)
        if not calib_path.is_absolute():
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                candidate = parent / calib_path
                if candidate.exists():
                    calib_path = candidate
                    break
        
        if not calib_path.exists():
            print(f"[WARNING] Calibration file not found: {calib_path}. Using full spectrum.")
            return 0, self.X.shape[1]
        
        # Load calibration table
        calib_df = pd.read_csv(calib_path)
        wavenums = calib_df['freq'].values  # Raman shift in cm⁻¹
        
        start_wavenum, end_wavenum = crop_region
        
        # Find closest indices
        start_idx = np.argmin(np.abs(wavenums - start_wavenum))
        end_idx = np.argmin(np.abs(wavenums - end_wavenum)) + 1  # +1 for slice end
        
        print(f"[OK] Spectral Crop: [{start_wavenum}, {end_wavenum}] cm-1 -> indices [{start_idx}, {end_idx}]")
        print(f"  Wavenumber mapping: {wavenums[start_idx]:.2f} -> {wavenums[end_idx-1]:.2f} cm-1")
        
        return start_idx, end_idx

    def apply_spectral_crop(self, crop_region: Optional[Tuple[float, float]] = None,
                           calib_file: Optional[str] = None) -> None:
        """
        Apply spectral cropping and renormalization in-place to X data.
        
        Important: Cropping changes the feature range, so renormalization is needed.
        Original normalization was to full spectrum's max (usually 2800-3000 cm⁻¹).
        After cropping to fingerprint region (700-1800), new max values emerge.
        
        Args:
            crop_region: Tuple of (start_cm1, end_cm1)
            calib_file: Path to RamanFreqTable.csv
        """
        start_idx, end_idx = self.get_wavenumber_indices(crop_region, calib_file)
        if start_idx == 0 and end_idx == self.X.shape[1]:
            print("  (No cropping applied)")
            return
        
        # Crop
        self.X = self.X[:, start_idx:end_idx]
        if self.wavenumbers is not None:
            self.wavenumbers = self.wavenumbers[start_idx:end_idx]
        
        # Renormalize to cropped region (critical for exp_103)
        print(f"  Renormalizing to cropped region...")
        max_per_spectrum = np.max(self.X, axis=1, keepdims=True)
        max_per_spectrum[max_per_spectrum == 0] = 1.0  # Avoid division by zero
        self.X = self.X / max_per_spectrum
        
        # Verify normalization
        min_val = np.min(self.X)
        max_val = np.max(self.X)
        print(f"    After renormalization: min={min_val:.6f}, max={max_val:.6f} (fingerprint region)")
        
        print(f"  Spectra shape after crop: {self.X.shape}")

    def apply_patient_cap(self, patient_cap: Optional[int] = None) -> None:
        """
        Downsample spectra to at most `patient_cap` per slide (patient).
        
        This prevents batch memorization when a single slide has many similar spectra
        (some slides have hundreds/thousands of correlated spectra).
        Each slide is from one patient with one tumor class.
        
        Args:
            patient_cap: Maximum spectra per slide (e.g., 500)
        
        Example:
            loader.apply_patient_cap(patient_cap=500)
        """
        if patient_cap is None:
            print("  (No patient cap applied)")
            return
        
        print(f"Applying patient cap: {patient_cap} spectra per slide...")
        
        # Use full group string (slide identifier) as the grouping key
        # Each group like "HF-448_V5B_1.h5_3" is a unique slide from one patient
        patient_ids = self.groups.copy()
        
        # Group by slide and downsample
        keep_indices = []
        for slide_id in np.unique(patient_ids):
            mask = (patient_ids == slide_id)
            indices = np.where(mask)[0]
            
            if len(indices) > patient_cap:
                # Randomly downsample to patient_cap
                keep = np.random.choice(indices, size=patient_cap, replace=False)
                keep_indices.extend(keep)
            else:
                # Keep all spectra from this slide
                keep_indices.extend(indices)
        
        keep_indices = np.sort(np.array(keep_indices))
        
        # Apply downsampling
        self.X = self.X[keep_indices]
        self.y_class = self.y_class[keep_indices]
        self.groups = self.groups[keep_indices]
        if self.y_tumor is not None:
            self.y_tumor = self.y_tumor[keep_indices]
        if self.coords is not None:
            self.coords = self.coords[keep_indices]

        print(f"[OK] Downsampled to {self.X.shape[0]} spectra (was {len(patient_ids)})")

    def __iter__(self):
        if self.split_file and self.split_file.exists():
            print(f"\n Using CV splits from: {self.split_file}")
            with open(self.split_file, 'r') as f:
                splits = json.load(f)
            
            # Extract cv_folds (new format) or use entire dict (old format)
            if "cv_folds" in splits:
                cv_splits = splits["cv_folds"]
                print(f"    Found 'cv_folds' section with {len(cv_splits)} folds")
            else:
                cv_splits = splits
                print(f"   ! Using old split format (consider migrating to 'cv_folds')")
            
            # If holdout_test_set exists in config, log it
            if self.config and 'cv_strategy' in self.config:
                if self.config['cv_strategy'].get('holdout_test_set'):
                    if 'test_set' in splits:
                        print(f"   ! Separate held-out test set exists: {len(splits['test_set'])} samples")
                        print(f"   !  These samples will NOT be used in CV training/validation")
            
            # Iterate through CV folds
            for i, (fold_name, split_data) in enumerate(cv_splits.items()):
                train_samples = split_data['train']
                test_samples = split_data['test']
                
                train_mask = np.isin(self.groups, train_samples)
                test_mask  = np.isin(self.groups, test_samples)

                print(f"   Fold {i}: {np.sum(train_mask)} train samples, {np.sum(test_mask)} val samples")

                yield {
                    'fold': i,
                    'fold_name': fold_name,
                    'train': (self.X[train_mask], self.y_class[train_mask]),
                    'val':   (self.X[test_mask],  self.y_class[test_mask]),
                    'train_sample_ids': np.unique(self.groups[train_mask]),
                    'val_sample_ids': np.unique(self.groups[test_mask]),
                    'train_groups': self.groups[train_mask],
                    'val_groups': self.groups[test_mask],
                    'train_coords': self.coords[train_mask] if self.coords is not None else None,
                    'val_coords': self.coords[test_mask] if self.coords is not None else None,
                }
        else:
            print("\n Split file not found or not configured.")
            print("   Falling back to StratifiedGroupKFold(n_splits=5)")
            print("   !  This may not match the intended CV design! Consider providing split_file.\n")
            sgkf = StratifiedGroupKFold(n_splits=5)
            for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(self.X, self.y_class, self.groups)):
                yield {
                    'fold': fold_idx,
                    'fold_name': f"fold_{fold_idx}",
                    'train': (self.X[train_idx], self.y_class[train_idx]),
                    'val':   (self.X[val_idx],   self.y_class[val_idx]),
                    'train_sample_ids': np.unique(self.groups[train_idx]),
                    'val_sample_ids': np.unique(self.groups[val_idx]),
                    'train_groups': self.groups[train_idx],
                    'val_groups': self.groups[val_idx],
                    'train_coords': self.coords[train_idx] if self.coords is not None else None,
                    'val_coords': self.coords[val_idx] if self.coords is not None else None,
                }

    def __len__(self):
        """Returns number of folds (default 5 if unknown)."""
        if self.split_file and self.split_file.exists():
            with open(self.split_file, 'r') as f:
                splits_data = json.load(f)
                # Handle new format with cv_folds section
                if "cv_folds" in splits_data:
                    return len(splits_data["cv_folds"])
                # Handle old format where folds are at top level
                return len(splits_data)
        return 5

    def close(self):
        if self.h5_file:
            self.h5_file.close()
