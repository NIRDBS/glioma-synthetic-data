# EXP 103+ Implementation Guide

## Overview

EXP 103 introduces flexible augmentation, spectral cropping, and patient-level downsampling to the glioma dataset pipeline. This guide explains how to use these features in your notebooks.

## 1. Spectral Cropping (Wavenumber-based)

### Problem Solved
- Scientists think in Raman shifts (cm⁻¹), not array indices
- Fingerprint region (700-1800 cm⁻¹) is most informative for tumor classification
- Manual index calculation is error-prone

### Solution
The loader now accepts Raman shifts in YAML and converts to indices automatically.

### In YAML Config (exp_103.yaml)
```yaml
preprocessing:
  crop_region: [700, 1800]  # Raman shifts in cm⁻¹
  calib_file: data/RamanFreqTable.csv  # Calibration table (required for mapping)
```

### In Notebook
```python
from src.utils.config_manager import ConfigManager
from src.data.loader import GliomaDataLoader

config = ConfigManager.load_config('exp_103')
loader = GliomaDataLoader.from_config(
    h5_data_path=config['paths']['processed_file'],
    config=config
)
# Cropping is applied automatically in from_config()

# Or manual apply:
# loader.apply_spectral_crop(crop_region=(700, 1800), 
#                           calib_file='data/RamanFreqTable.csv')

print(f"X shape after crop: {loader.X.shape}")  # e.g., (50000, 380)
```

### How It Works
1. Reads RamanFreqTable.csv (idx, freq pairs)
2. Finds closest array indices to requested Raman shifts
3. Slices X[:, start_idx:end_idx]
4. All subsequent analysis uses cropped spectra only

---

## 2. Patient-Level Downsampling

### Problem Solved
- Some slides have 2000+ very similar spectra from same region
- Leads to batch memorization and overfitting
- Other slides have only 50 spectra
- Imbalance causes unfair training

### Solution
Cap spectra per patient per class to a maximum (e.g., 500).

### In YAML Config
```yaml
preprocessing:
  patient_cap: 500  # Max spectra per patient per class
```

### In Notebook
```python
# Applied automatically in from_config()
# Or manual:
# loader.apply_patient_cap(patient_cap=500)

print(f"Spectra before cap: 150000")
print(f"Spectra after cap: {loader.X.shape[0]}")  # e.g., 47000
```

### How It Works
1. Groups spectra by (patient_id, class)
2. For groups with > patient_cap spectra: randomly keep patient_cap
3. Keep all groups with ≤ patient_cap spectra
4. Concatenate all kept indices
5. Result: balanced dataset without memorization

---

## 3. Configurable Augmentation

### Problem Solved
- Augmentation shouldn't be hardcoded
- Different experiments need different augmentation strategies
- Augmentation only happens during training, not inference

### Solution
Define augmentation in YAML, use RamanAugmentationGenerator for training.

### In YAML Config
```yaml
augmentation:
  enabled: true
  max_shift: 3           # ±3 pixel wavenumber shift
  mixup_alpha: 0.2       # Beta distribution parameter
```

### For CVAE Pre-training

```python
from src.data.augmentation import RamanAugmentationGenerator, get_augmentation_config

# Extract augmentation config
aug_cfg = get_augmentation_config(config)

if aug_cfg:
    # Create training generator with augmentation
    train_gen = RamanAugmentationGenerator(
        X_train, y_train,
        batch_size=256,
        **aug_cfg
    )
    
    # Create validation generator WITHOUT augmentation
    val_gen = RamanAugmentationGenerator(
        X_val, y_val,
        batch_size=256,
        augment=False  # No augmentation during validation
    )
    
    cvae.fit(train_gen, validation_data=val_gen, epochs=70)
else:
    # Fallback: no augmentation
    cvae.fit(X_train, X_train, validation_split=0.1, epochs=70)
```

### For Classifier (TATR: Train-Augmented-Train-Real)

```python
from src.data.augmentation import RamanAugmentationGenerator, get_augmentation_config

aug_cfg = get_augmentation_config(config)

if aug_cfg:
    # Generate one epoch of augmented data
    aug_gen = RamanAugmentationGenerator(
        X_train, y_train,
        batch_size=32,
        **aug_cfg,
        augment=True
    )
    
    X_aug_list = []
    for batch_x, _ in aug_gen:
        X_aug_list.append(batch_x)
    
    X_aug = np.vstack(X_aug_list)  # All augmented samples
    X_combined = np.vstack([X_train, X_aug])  # Real + synthetic mix
    y_combined = np.concatenate([y_train, y_train])
    
    # Train classifiers on combined data
    rf.fit(X_combined, y_combined)
    xgb_model.fit(X_combined, y_combined)
else:
    # Fallback: train on real data only
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
```

---

## 4. CVAE with Spectral Loss

### Problem Solved
- Standard MSE only cares about pixel-level errors
- Ignores spectral shape preservation
- Reconstructed spectra can look different despite low MSE

### Solution
Combine MSE + Cosine Similarity in loss function.

### In YAML Config
```yaml
models:
  cvae:
    loss_function: "mse_cosine_combo"
    gamma: 0.7  # MSE weight
    delta: 0.3  # Cosine similarity weight
```

### In Notebook
```python
from src.data.augmentation import spectral_loss
import tensorflow as tf

# Option 1: Use predefined loss
def custom_loss(y_true, y_pred):
    return spectral_loss(
        y_true, y_pred,
        mse_weight=config['models']['cvae'].get('gamma', 0.7),
        cosine_weight=config['models']['cvae'].get('delta', 0.3)
    )

cvae.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss)

# Option 2: Manual implementation in call()
```

---

## 5. Data Leakage Control at Training Time

### Where Augmentation Happens
- **Training only**: RamanAugmentationGenerator applies shifts + mixup
- **Validation**: Generator with augment=False returns unaugmented batches
- **Test**: No augmentation

### Intra-Class Mixup Design
- Only mixes spectra from **same tumor type** (same class)
- If classes differ: no blending (λ=1.0)
- No label mixup: y labels stay pure (not blurred)
- Validates: λ ~ Beta(α, α), ensures convex combinations

### Patient-Level Assurance
- Spectra from same patient are grouped
- Patient cap applied **per class**, so tumor/healthy not mixed
- Cross-fold: train/val split from config ensures no patient overlap

---

## 6. Backward Compatibility

### Old Experiments (exp_101, exp_102)
Still work without changes:

```python
# exp_101 (no augmentation, no cropping, no patient cap)
config = ConfigManager.load_config('exp_101')
loader = GliomaDataLoader.from_config(
    h5_data_path=config['paths']['processed_file'],
    config=config
)
# No crop_region, patient_cap, or augmentation in config → all skipped
```

### New Experiments (exp_103+)
Opt-in to new features via YAML:

```yaml
# exp_103
preprocessing:
  crop_region: [700, 1800]
  patient_cap: 500
augmentation:
  enabled: true
```

---

## 7. Complete Notebook Template (exp_103)

### Setup
```python
from src.utils.config_manager import ConfigManager
from src.data.loader import GliomaDataLoader
from src.data.augmentation import RamanAugmentationGenerator, spectral_loss
import numpy as np

config = ConfigManager.load_config('exp_103')
loader = GliomaDataLoader.from_config(
    h5_data_path=config['paths']['processed_file'],
    config=config
)
```

### CVAE Training
```python
train_gen = RamanAugmentationGenerator(
    X_train, y_train,
    batch_size=256,
    max_shift=3,
    mixup_alpha=0.2,
    augment=True
)

val_gen = RamanAugmentationGenerator(
    X_val, y_val,
    batch_size=256,
    augment=False
)

def custom_loss(y_true, y_pred):
    return spectral_loss(y_true, y_pred, mse_weight=0.7, cosine_weight=0.3)

cvae.compile(optimizer='adam', loss=custom_loss)
cvae.fit(train_gen, validation_data=val_gen, epochs=70)
```

### Classifier (TATR)
```python
aug_gen = RamanAugmentationGenerator(X_train, y_train, augment=True)
X_aug = np.vstack([batch_x for batch_x, _ in aug_gen])
X_combined = np.vstack([X_train, X_aug])
y_combined = np.concatenate([y_train, y_train])

rf.fit(X_combined, y_combined)
```

---

## Summary of Changes

| Feature | File | Change | Config Key |
|---------|------|--------|-----------|
| Spectral Crop | loader.py | Added `apply_spectral_crop()` | `crop_region` |
| Patient Cap | loader.py | Added `apply_patient_cap()` | `patient_cap` |
| Augmentation | augmentation.py | New `RamanAugmentationGenerator` | `augmentation.{enabled, max_shift, mixup_alpha}` |
| Spectral Loss | augmentation.py | New `spectral_loss()` | `loss_function` |
| Auto-apply | loader.py | `from_config()` now calls crop/cap | All preprocessing.* |

---

## Troubleshooting

### "Calibration file not found"
- Ensure `calib_file` path in config is correct
- Check that RamanFreqTable.csv exists
- Path can be relative (resolved from project root) or absolute

### "crop_region not applied"
- Check config has `preprocessing.crop_region` defined
- Verify `from_config()` is used (not manual `__init__`)
- Manual: call `loader.apply_spectral_crop(crop_region=...)`

### "Augmentation not working"
- Check `augmentation.enabled: true` in config
- Verify RamanAugmentationGenerator receives augment=True
- Validation/test sets should use augment=False

### "Data leakage concerns"
- Intra-class mixup only blends same-class spectra
- Patient cap applied per (patient, class) group
- Train/val split from config ensures no patient overlap
