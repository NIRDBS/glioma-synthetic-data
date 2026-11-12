"""
On-the-fly spectral augmentation for Raman spectra.

Supports:
1. Spectral shift (calibration drift simulation via cubic interpolation)
   - Real-valued shifts (e.g., 0.5 pixels) handled by cubic interpolation
2. Within-class mixup (linear blending of spectra within same class)
   - Randomly draw 2 spectra from same class, blend with p ~ Uniform[0, mixup_alpha]
3. Batched augmentation with on-the-fly application

Used in exp_103+ for CVAE pre-training with full augmentation.
Default: no augmentation (safe for exp_101, exp_102 compatibility).
"""

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from typing import Optional, Tuple


def spectral_shift(spectrum, shift_amount, kind='cubic'):
    """
    Apply spectral shift via cubic interpolation.
    
    Simulates calibration drift by shifting the spectrum along the wavenumber axis.
    Uses cubic interpolation to handle fractional pixel shifts (e.g., 0.5 pixels).
    
    Args:
        spectrum: (n_features,) array - single spectrum
        shift_amount: float - shift in pixels (can be fractional, e.g., 0.5, -1.2)
        kind: str - interpolation kind ('cubic', 'quadratic', 'linear')
    
    Returns:
        shifted_spectrum: (n_features,) array - spectrum after shift
    """
    n = len(spectrum)
    x_orig = np.arange(n, dtype=np.float32)
    
    # Build cubic interpolator (extrapolate at edges)
    f = interp1d(x_orig, spectrum, kind=kind, bounds_error=False, fill_value='extrapolate')
    
    # New positions after shift
    x_shifted = x_orig - shift_amount
    
    # Interpolate at new positions
    spectrum_shifted = f(x_shifted)
    
    return spectrum_shifted.astype(np.float32)


def baseline_wobble(spectrum, max_amplitude=0.05, n_control_points=5):
    """
    Add a random smooth baseline (low-order spline through a few random control
    points), making the classifier invariant to residual baseline error left
    over from imperfect correction (docs/MIL_REFRAME_PLAN.md §7.2 -- "the most
    valuable of these", directly motivated by the Stage 0 baseline study).

    Args:
        spectrum: (n_features,) array, already baseline-corrected + normalised
        max_amplitude: float - wobble amplitude as a fraction of the spectrum's
            own dynamic range (so it scales with the data, not a fixed unit)
        n_control_points: int - fewer points -> smoother/broader wobble

    Returns:
        (n_features,) array with an added smooth perturbation
    """
    n = len(spectrum)
    control_x = np.linspace(0, n - 1, n_control_points)
    scale = max_amplitude * (np.ptp(spectrum) + 1e-8)
    control_y = np.random.uniform(-scale, scale, size=n_control_points)
    f = interp1d(control_x, control_y, kind='cubic', bounds_error=False, fill_value='extrapolate')
    wobble = f(np.arange(n))
    return (spectrum + wobble).astype(np.float32)


def intensity_jitter(spectrum, scale_range=(0.9, 1.1)):
    """
    Multiplicative intensity scaling -- simulates laser power / focus variation
    between acquisitions. A single scalar per spectrum (not per band), since
    the physical cause is a global gain change, not a band-specific effect.
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return (spectrum * scale).astype(np.float32)


def additive_noise(spectrum, snr_db=30.0):
    """
    Additive Gaussian noise at a target SNR (dB), computed from the spectrum's
    own signal power so the noise level scales with the data rather than being
    a fixed absolute unit.
    """
    signal_power = np.mean(spectrum ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=spectrum.shape)
    return (spectrum + noise).astype(np.float32)


def band_dropout(spectrum, max_width=10, n_bands=1, fill='local_mean'):
    """
    Zero out (or locally-flatten) a small contiguous band -- simulates detector
    defects / saturated regions. ``fill='local_mean'`` avoids injecting a sharp,
    unrealistic zero-notch that would itself become a spurious learnable
    feature; it replaces the dropped band with its own pre-drop mean instead.
    """
    spectrum = spectrum.copy()
    n = len(spectrum)
    for _ in range(n_bands):
        width = np.random.randint(1, max_width + 1)
        start = np.random.randint(0, max(n - width, 1))
        end = start + width
        fill_value = spectrum[start:end].mean() if fill == 'local_mean' else 0.0
        spectrum[start:end] = fill_value
    return spectrum.astype(np.float32)


def within_class_mixup(spectrum, X_data, y_data, class_idx, mixup_alpha):
    """
    Randomly blend current spectrum with another from the same class.
    
    Implements: new_spectrum = p * other_spectrum + (1-p) * current_spectrum
    where p is randomly drawn from Uniform[0, mixup_alpha].
    
    Args:
        spectrum: (n_features,) array - current spectrum to augment
        X_data: (n_samples, n_features) array - all training spectra
        y_data: (n_samples,) array - all training labels
        class_idx: int - class label of current spectrum
        mixup_alpha: float - upper bound for random p
    
    Returns:
        mixed_spectrum: (n_features,) array - augmented spectrum
    """
    # Find all indices with same class
    same_class_indices = np.where(y_data == class_idx)[0]
    
    # Randomly pick another spectrum from same class
    other_idx = np.random.choice(same_class_indices)
    other_spectrum = X_data[other_idx]
    
    # Random blending coefficient p in [0, mixup_alpha]
    p = np.random.uniform(0, mixup_alpha)
    
    # Linear combination
    mixed_spectrum = p * other_spectrum + (1 - p) * spectrum
    
    return mixed_spectrum.astype(np.float32)


class AugmentedDataGenerator:
    """
    Wraps preprocessed data and applies on-the-fly augmentation.
    
    Creates augmented batches without storing augmented data in memory.
    Supports both full augmentation (shift + mixup) and individual augmentations.
    
    Augmentation strategy:
    1. Spectral shift: Real-valued cubic interpolation (handles 0.5 pixel shifts)
    2. Within-class mixup: Randomly draw 2 spectra from same class, blend with p ~ Uniform[0, mixup_alpha]
    
    Default: Disabled (safe for exp_101, exp_102 backward compatibility).
    
    Usage:
        gen = AugmentedDataGenerator(X_train, y_train, augmentation_config)
        train_dataset = gen.get_tf_dataset(batch_size=256, epochs=70)
        cvae.fit(train_dataset, epochs=70)
    """
    
    def __init__(self, X_data, y_data, augmentation_config=None):
        """
        Initialize augmentation generator.
        
        Args:
            X_data: (n_samples, n_features) or (n_samples, n_features, 1) - training spectra
            y_data: (n_samples,) - training labels
            augmentation_config: dict with keys:
                - 'enabled': bool - whether to enable augmentation (default: False)
                - 'max_shift': float - max spectral shift in pixels (default: 3.0)
                - 'mixup_alpha': float - upper bound for mixup blending (default: 0.5)
                - 'baseline_wobble': {'enabled': bool, 'max_amplitude': float} --
                  default enabled (docs/MIL_REFRAME_PLAN.md §7.2: "the most valuable
                  of these"), directly motivated by the Stage 0 baseline study
                - 'intensity_jitter': {'enabled': bool, 'scale_range': (lo, hi)}
                - 'additive_noise': {'enabled': bool, 'snr_db': float}
                - 'band_dropout': {'enabled': bool, 'max_width': int, 'n_bands': int}
                  (the last three default disabled -- opt in per experiment)
        """
        # Handle 3D input (n_samples, n_features, 1) -> (n_samples, n_features)
        if X_data.ndim == 3:
            X_data = X_data.squeeze(axis=-1)

        self.X_data = X_data.astype(np.float32)
        self.y_data = y_data.astype(np.int32)

        # Default config: no augmentation (backward compatible)
        if augmentation_config is None:
            augmentation_config = {'enabled': False}

        self.config = augmentation_config
        self.enabled = augmentation_config.get('enabled', False)
        self.max_shift = augmentation_config.get('max_shift', 3.0)
        self.mixup_alpha = augmentation_config.get('mixup_alpha', 0.5)

        self.baseline_wobble_cfg = augmentation_config.get('baseline_wobble', {'enabled': True})
        self.intensity_jitter_cfg = augmentation_config.get('intensity_jitter', {'enabled': False})
        self.additive_noise_cfg = augmentation_config.get('additive_noise', {'enabled': False})
        self.band_dropout_cfg = augmentation_config.get('band_dropout', {'enabled': False})

        self.n_samples = len(X_data)
        self.n_features = X_data.shape[1]

    def augment_sample(self, idx):
        """
        Augment a single sample at index idx.

        Applies in order: within-class mixup, spectral shift, then whichever of
        baseline wobble / intensity jitter / additive noise / band dropout are
        enabled (each independently switchable, see __init__'s docstring).
        Order matters only a little here -- wobble/jitter/noise/dropout each
        perturb the spectrum independently and are applied after the two
        structural augmentations (mixup, shift) that combine two spectra or
        move the whole signal, so the later ones perturb something already
        spectrum-shaped rather than a blend-in-progress.

        Args:
            idx: int - index of sample to augment

        Returns:
            augmented_spectrum: (n_features,) augmented spectrum
        """
        if not self.enabled:
            return self.X_data[idx]

        spectrum = self.X_data[idx].copy()
        class_idx = self.y_data[idx]

        # 1. Within-class mixup
        spectrum = within_class_mixup(
            spectrum, self.X_data, self.y_data,
            class_idx, self.mixup_alpha
        )

        # 2. Spectral shift (random fractional shift)
        shift = np.random.uniform(-self.max_shift, self.max_shift)
        spectrum = spectral_shift(spectrum, shift)

        # 3. Baseline wobble
        if self.baseline_wobble_cfg.get('enabled', True):
            spectrum = baseline_wobble(
                spectrum, max_amplitude=self.baseline_wobble_cfg.get('max_amplitude', 0.05))

        # 4. Intensity jitter
        if self.intensity_jitter_cfg.get('enabled', False):
            spectrum = intensity_jitter(
                spectrum, scale_range=self.intensity_jitter_cfg.get('scale_range', (0.9, 1.1)))

        # 5. Additive noise
        if self.additive_noise_cfg.get('enabled', False):
            spectrum = additive_noise(spectrum, snr_db=self.additive_noise_cfg.get('snr_db', 30.0))

        # 6. Band dropout
        if self.band_dropout_cfg.get('enabled', False):
            spectrum = band_dropout(
                spectrum, max_width=self.band_dropout_cfg.get('max_width', 10),
                n_bands=self.band_dropout_cfg.get('n_bands', 1))

        return spectrum
    
    def get_batches(self, batch_size=256, epochs=1, shuffle=True):
        """
        Generator that yields augmented batches.
        
        Args:
            batch_size: int - batch size
            epochs: int - number of epochs to iterate
            shuffle: bool - whether to shuffle indices each epoch
        
        Yields:
            (X_batch, y_batch): tuple of augmented batch data and labels
        """
        for epoch in range(epochs):
            indices = np.arange(self.n_samples)
            if shuffle:
                np.random.shuffle(indices)
            
            for batch_start in range(0, self.n_samples, batch_size):
                batch_end = min(batch_start + batch_size, self.n_samples)
                batch_indices = indices[batch_start:batch_end]
                
                # Augment each sample in batch
                X_batch = np.array([
                    self.augment_sample(idx) for idx in batch_indices
                ], dtype=np.float32)
                y_batch = self.y_data[batch_indices].astype(np.int32)
                
                yield X_batch, y_batch
    
    def get_tf_dataset(self, batch_size=256, epochs=1, shuffle=True):
        """
        Wrap batches as TensorFlow Dataset.
        
        Args:
            batch_size: int - batch size
            epochs: int - number of epochs
            shuffle: bool - whether to shuffle
        
        Returns:
            tf.data.Dataset - yields (X_batch, y_batch) tuples
        """
        def generator():
            for X_batch, y_batch in self.get_batches(batch_size, epochs, shuffle):
                yield X_batch, y_batch
        
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, self.n_features), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )


# Backward compatibility alias
class RamanAugmentationGenerator(AugmentedDataGenerator):
    """Backward compatibility alias for AugmentedDataGenerator."""
    pass


def spectral_loss(y_true, y_pred, mse_weight: float = 0.7, cosine_weight: float = 0.3):
    """
    Combined MSE + Cosine Similarity loss for CVAE pre-training on spectral data.
    
    Combines:
    - MSE (pixel-level reconstruction error)
    - Cosine similarity (spectral shape preservation)
    
    For reconstructing Raman spectra, this encourages both accurate pixel values
    AND preservation of spectral shape/features.
    
    Args:
        y_true: Target spectra (batch_size, n_features)
        y_pred: Reconstructed spectra (batch_size, n_features)
        mse_weight: Weight for MSE term
        cosine_weight: Weight for cosine similarity term
    
    Returns:
        Scalar loss value
    
    Example (in CVAE model):
        def custom_loss(y_true, y_pred):
            return spectral_loss(y_true, y_pred, mse_weight=0.7, cosine_weight=0.3)
        
        cvae.compile(optimizer='adam', loss=custom_loss)
    """
    import tensorflow as tf
    
    # MSE: pixel-level reconstruction
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Cosine similarity: spectral shape (normalize vectors first)
    y_true_norm = tf.nn.l2_normalize(y_true, axis=1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=1)
    
    # Cosine similarity in [-1, 1], convert to loss in [0, 2]
    cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=1)
    cosine_loss = 1.0 - cosine_sim  # Convert similarity to loss
    
    # Combine
    combined_loss = mse_weight * mse + cosine_weight * tf.reduce_mean(cosine_loss)
    
    return combined_loss


def get_augmentation_config(config_dict: dict) -> dict:
    """
    Extract augmentation configuration from YAML config.
    
    Args:
        config_dict: Configuration dictionary (from ConfigManager)
    
    Returns:
        Dictionary with augmentation parameters, or empty dict if augmentation disabled
    
    Example:
        aug_cfg = get_augmentation_config(config)
        if aug_cfg:
            train_gen = RamanAugmentationGenerator(X, y, **aug_cfg)
    """
    aug_cfg = config_dict.get('augmentation', {})

    if not aug_cfg.get('enabled', False):
        return {}

    return {
        'enabled': True,
        'max_shift': aug_cfg.get('max_shift', 3),
        'mixup_alpha': aug_cfg.get('mixup_alpha', 0.2),
        'baseline_wobble': aug_cfg.get('baseline_wobble', {'enabled': True}),
        'intensity_jitter': aug_cfg.get('intensity_jitter', {'enabled': False}),
        'additive_noise': aug_cfg.get('additive_noise', {'enabled': False}),
        'band_dropout': aug_cfg.get('band_dropout', {'enabled': False}),
    }


# ============================================================================
# DYNAMIC PATIENT CAP SAMPLING
# ============================================================================

class DynamicPatientCapSequence:
    """
    Dynamic patient-level downsampling for training.
    
    Instead of permanently downsampling during preprocessing, this generator
    samples ≤N spectra per (patient_id, class) on each epoch, providing:
    
    - Data efficiency: Uses all available data (variable subsets per epoch)
    - Implicit regularization: Different samples each epoch prevents overfitting
    - Class balance: Fair representation regardless of patient slide count
    
    Key insight: Each epoch, the model sees a DIFFERENT random 500-sample subset
    from each patient, effectively acting as data augmentation through sampling.
    
    Typical patient imbalance:
        Before: Class_A Patient #5 has 1500 spectra, Patient #10 has 100
        After: Each epoch, both sampled to min(N, 500) → balanced training
    
    Note: Implements Keras Sequence interface without explicit inheritance.
    Compatible with model.fit(generator) due to duck typing.
    
    Usage:
        train_gen = DynamicPatientCapSequence(
            X_train, y_train, patient_ids=patient_ids,
            patient_cap=500, batch_size=256
        )
        model.fit(train_gen, validation_data=(X_val, y_val), epochs=70)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray,
                 patient_cap: int = 500, batch_size: int = 256, seed: int = 42):
        """
        Initialize dynamic patient cap generator.
        
        Args:
            X: (n_samples, n_features) spectra array (FULL, uncapped)
            y: (n_samples,) class labels
            patient_ids: (n_samples,) patient identifiers (used for grouping)
            patient_cap: Max spectra per (patient_id, class) to sample per epoch
            batch_size: Batch size for iteration
            seed: Random seed for reproducibility
        """
        self.X_full = X.astype(np.float32)
        self.y_full = y.astype(np.int32)
        self.patient_ids = np.array(patient_ids)
        self.patient_cap = patient_cap
        self.batch_size = batch_size
        self.seed = seed
        
        np.random.seed(seed)
        
        # Pre-compute (patient_id, class) groups and their indices
        self.groups = {}  # {(patient_id, class): [idx1, idx2, ...]}
        for idx in range(len(self.X_full)):
            group_key = (self.patient_ids[idx], self.y_full[idx])
            if group_key not in self.groups:
                self.groups[group_key] = []
            self.groups[group_key].append(idx)
        
        # Sample initial epoch
        self.sampled_indices = None
        self.on_epoch_end()
    
    def _sample_epoch(self) -> np.ndarray:
        """
        Sample up to patient_cap spectra from each (patient_id, class) group.
        
        Returns:
            Array of sampled indices (total may vary slightly due to group sizes)
        """
        sampled = []
        for group_indices in self.groups.values():
            # Sample up to patient_cap from this group (or all if fewer)
            n_sample = min(len(group_indices), self.patient_cap)
            sampled_group = np.random.choice(group_indices, size=n_sample, replace=False)
            sampled.extend(sampled_group)
        
        sampled_indices = np.array(sampled)
        np.random.shuffle(sampled_indices)
        return sampled_indices
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        # Total sampled size depends on group distribution
        # Approximate: min(total_spectra, n_groups * patient_cap)
        approx_size = sum(min(len(group), self.patient_cap) 
                         for group in self.groups.values())
        return int(np.ceil(approx_size / self.batch_size))
    
    def on_epoch_end(self) -> None:
        """Resample at end of epoch (different 500-sample subset next epoch)."""
        self.sampled_indices = self._sample_epoch()
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batch from sampled epoch data.
        
        Args:
            index: Batch index
        
        Returns:
            Tuple of (batch_x, batch_y)
        """
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.sampled_indices))
        
        batch_indices = self.sampled_indices[start_idx:end_idx]
        batch_x = self.X_full[batch_indices].astype(np.float32)
        batch_y = self.y_full[batch_indices]
        
        return batch_x, batch_y
    
    def get_statistics(self) -> dict:
        """
        Return statistics about sampling.
        
        Returns:
            Dictionary with sampling info for debugging
        """
        stats = {
            'n_groups': len(self.groups),
            'total_spectra_full': len(self.X_full),
            'patient_cap': self.patient_cap,
            'n_unique_patients': len(np.unique(self.patient_ids)),
            'n_classes': len(np.unique(self.y_full)),
        }
        
        # Per-group stats
        group_sizes_before = [len(g) for g in self.groups.values()]
        group_sizes_after = [min(len(g), self.patient_cap) for g in self.groups.values()]
        
        stats.update({
            'group_sizes_before_min': min(group_sizes_before),
            'group_sizes_before_max': max(group_sizes_before),
            'group_sizes_before_mean': np.mean(group_sizes_before),
            'group_sizes_after_min': min(group_sizes_after),
            'group_sizes_after_max': max(group_sizes_after),
            'group_sizes_after_mean': np.mean(group_sizes_after),
            'total_spectra_sampled': sum(group_sizes_after),
            'reduction_pct': 100 * (1 - sum(group_sizes_after) / sum(group_sizes_before)),
        })
        
        return stats


def get_patient_cap_config(config_dict: dict) -> dict:
    """
    Extract patient cap configuration from cv_strategy section.
    
    Args:
        config_dict: Configuration dictionary (from ConfigManager)
    
    Returns:
        Dictionary with patient_cap and cap_strategy, or empty dict if not configured
    
    Example:
        cap_cfg = get_patient_cap_config(config)
        if cap_cfg:
            train_gen = DynamicPatientCapSequence(X_train, y_train, 
                                                  patient_ids=patient_ids, 
                                                  **cap_cfg)
    """
    cv_cfg = config_dict.get('cv_strategy', {})
    
    patient_cap = cv_cfg.get('patient_cap', None)
    cap_strategy = cv_cfg.get('cap_strategy', 'dynamic_per_epoch')
    
    if patient_cap is None:
        return {}
    
    return {
        'patient_cap': patient_cap,
        'cap_strategy': cap_strategy,
    }
