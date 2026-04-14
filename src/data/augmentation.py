"""
Raman Spectra Augmentation Module
==================================

Provides data augmentation for Raman spectroscopy:
- Spectral shifting (calibration drift simulation)
- Intra-class mixup (convex combinations of same-class spectra)
- Training-time augmentation via Keras Sequence

Used in exp_103+ for:
1. CVAE pre-training: RamanAugmentationGenerator feeds augmented data
2. TATR (Trains-Augmented-Train-Real): Classic models trained on real + augmented mix
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple


class RamanAugmentationGenerator(tf.keras.utils.Sequence):
    """
    Training-time augmentation for Raman spectroscopy.
    
    Features:
    - Spectral shifting: simulates calibration drift (±N wavenumber pixels)
    - Intra-class mixup: convex combinations of spectra from the same class
    - Configurable enable/disable for augmentation during inference
    - Stratified shuffling to maintain class balance
    
    Usage (CVAE pre-training):
        train_gen = RamanAugmentationGenerator(X_train, y_train, batch_size=256,
                                              max_shift=3, mixup_alpha=0.2)
        cvae.fit(train_gen, validation_data=val_gen, epochs=70)
    
    Usage (TATR for classifiers):
        aug_gen = RamanAugmentationGenerator(X_train, y_train, batch_size=32,
                                            max_shift=3, augment=True)
        # Draw one epoch of augmented data
        for batch_x, _ in aug_gen:
            X_aug = np.vstack([batch_x for batch_x, _ in aug_gen])  # Single epoch
        X_combined = np.vstack([X_train, X_aug])
        y_combined = np.concatenate([y_train, y_train])  # Labels repeat
        rf.fit(X_combined, y_combined)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 256,
                 max_shift: int = 3, mixup_alpha: float = 0.2,
                 augment: bool = True, seed: int = 42):
        """
        Initialize augmentation generator.
        
        Args:
            X: (n_samples, n_features) spectra array
            y: (n_samples,) class labels
            batch_size: Batch size for iteration
            max_shift: Max wavenumber pixel shift (±N)
            mixup_alpha: Beta distribution parameter for mixup (higher = more aggressive)
            augment: If False, returns unaugmented batches (for inference)
            seed: Random seed for reproducibility
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)
        self.batch_size = batch_size
        self.max_shift = max_shift
        self.mixup_alpha = mixup_alpha
        self.augment = augment
        
        np.random.seed(seed)
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self) -> None:
        """Shuffle indices at end of epoch."""
        np.random.shuffle(self.indices)

    def _shift_spectra(self, batch_x: np.ndarray) -> np.ndarray:
        """
        Spectral shift augmentation: simulates calibration drift.
        
        Randomly shifts spectra left/right by up to `max_shift` pixels,
        padding edges with the boundary value.
        
        Args:
            batch_x: (batch_size, n_features) spectra
        
        Returns:
            Shifted spectra of same shape
        """
        shifted_x = np.zeros_like(batch_x)
        for i in range(len(batch_x)):
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            if shift > 0:
                # Shift right: copy data right, pad left with edge value
                shifted_x[i, shift:] = batch_x[i, :-shift]
                shifted_x[i, :shift] = batch_x[i, 0]
            elif shift < 0:
                # Shift left: copy data left, pad right with edge value
                shifted_x[i, :shift] = batch_x[i, -shift:]
                shifted_x[i, shift:] = batch_x[i, -1]
            else:
                # No shift
                shifted_x[i] = batch_x[i]
        return shifted_x

    def _intra_class_mixup(self, batch_x: np.ndarray, batch_y: np.ndarray) -> np.ndarray:
        """
        Intra-class mixup: convex combinations of same-class spectra.
        
        Only blends spectra from the same tumor type/class. If classes don't match,
        no blending occurs (lam=1.0, identity operation).
        
        Mathematical:
            x_mixed[i] = λ * x[i] + (1 - λ) * x[perm[i]]
            where λ ~ Beta(α, α)
        
        Args:
            batch_x: (batch_size, n_features) spectra
            batch_y: (batch_size,) class labels
        
        Returns:
            Mixed spectra of same shape (with same y labels, no label mixup)
        """
        # Shuffle batch to create permutation
        shuffle_idx = np.random.permutation(len(batch_x))
        batch_x_shuffled = batch_x[shuffle_idx]
        batch_y_shuffled = batch_y[shuffle_idx]
        
        # Draw lambda from Beta distribution
        # Shape (batch_size, 1) for broadcasting
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, 
                            size=(len(batch_x), 1)).astype(np.float32)
        
        # Only blend if classes match, else lambda=1 (no change)
        same_class_mask = (batch_y == batch_y_shuffled).reshape(-1, 1).astype(np.float32)
        lam = same_class_mask * lam + (1 - same_class_mask) * 1.0
        
        # Blend spectra
        mixed_x = lam * batch_x + (1 - lam) * batch_x_shuffled
        
        return mixed_x.astype(np.float32)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get augmented batch.
        
        Args:
            index: Batch index
        
        Returns:
            Tuple of (batch_x, batch_y)
            - batch_x: (batch_size, n_features) augmented spectra or originals
            - batch_y: (batch_size,) class labels (unchanged)
        """
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.X))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load batch
        batch_x = self.X[batch_indices].copy().astype(np.float32)
        batch_y = self.y[batch_indices]
        
        # Apply augmentation if enabled
        if self.augment:
            # Apply spectral shift first
            batch_x = self._shift_spectra(batch_x)
            # Then apply intra-class mixup
            batch_x = self._intra_class_mixup(batch_x, batch_y)
        
        return batch_x, batch_y


def spectral_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
                  mse_weight: float = 0.7, cosine_weight: float = 0.3) -> tf.Tensor:
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
        'max_shift': aug_cfg.get('max_shift', 3),
        'mixup_alpha': aug_cfg.get('mixup_alpha', 0.2),
        'augment': True
    }
