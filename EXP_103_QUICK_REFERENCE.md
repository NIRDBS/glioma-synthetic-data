# EXP 103+ Quick Reference

One-page reference for using exp_103 features in notebooks.

## TL;DR - The Changes

| Feature | Config Key | What | Where |
|---------|-----------|------|-------|
| Spectral Crop | `preprocessing.crop_region` | 700-1800 cm⁻¹ → indices | GliomaDataLoader.apply_spectral_crop() |
| Patient Cap | `preprocessing.patient_cap` | Max 500/patient/class | GliomaDataLoader.apply_patient_cap() |
| Augment Shift | `augmentation.max_shift` | ±3 pixel shift | RamanAugmentationGenerator._shift_spectra() |
| Augment Mixup | `augmentation.mixup_alpha` | Convex mix (same-class) | RamanAugmentationGenerator._intra_class_mixup() |
| Spectral Loss | `models.cvae.loss_function` | MSE + cosine | spectral_loss() |

---

## 1-Minute Setup

### Add to Notebooks
```python
from src.data.augmentation import RamanAugmentationGenerator, spectral_loss, get_augmentation_config
```

### In exp_103.yaml
```yaml
crop_region: [700, 1800]
patient_cap: 500
augmentation:
  enabled: true
  max_shift: 3
  mixup_alpha: 0.2
models:
  cvae:
    loss_function: "mse_cosine_combo"
    gamma: 0.7
    delta: 0.3
```

---

## Copy-Paste Blocks

### Block 1: Load Data with Preprocessing
```python
from src.data.loader import GliomaDataLoader
from src.utils.config_manager import ConfigManager

config = ConfigManager.load_config('exp_103')
loader = GliomaDataLoader.from_config(
    h5_data_path=config['paths']['processed_file'],
    config=config
)
# Auto-applies crop_region and patient_cap from config
```

### Block 2: TATR for RF/XGBoost
```python
from src.data.augmentation import get_augmentation_config, RamanAugmentationGenerator
import numpy as np

aug_cfg = get_augmentation_config(config)
if aug_cfg:
    aug_gen = RamanAugmentationGenerator(X_train, y_train, batch_size=32, **aug_cfg, augment=True)
    X_aug = np.vstack([batch_x for batch_x, _ in aug_gen])
    X_train = np.vstack([X_train, X_aug])
    y_train = np.concatenate([y_train, y_train])

# Train on combined real+augmented
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
```

### Block 3: Augmented CVAE
```python
from src.data.augmentation import RamanAugmentationGenerator, spectral_loss, get_augmentation_config

aug_cfg = get_augmentation_config(config)

# Create generators
if aug_cfg:
    train_gen = RamanAugmentationGenerator(X_train, y_train, batch_size=256, **aug_cfg, augment=True)
    val_gen = RamanAugmentationGenerator(X_val, y_val, batch_size=256, augment=False)
else:
    train_gen, val_gen = None, None

# Compile with spectral loss
cvae.compile(
    optimizer='adam',
    loss=lambda y_t, y_p: spectral_loss(y_t, y_p, mse_weight=0.7, cosine_weight=0.3)
)

# Train
cvae.fit(
    train_gen or X_train,
    validation_data=(val_gen or X_val),
    epochs=70
)
```

### Block 4: Test/Predict (NO AUGMENTATION)
```python
# Define inference generator with augment=False
test_gen = RamanAugmentationGenerator(X_test, y_test, augment=False)

# Or use raw data
preds = model.predict(X_test)  # Direct, no augmentation
```

---

## Common Patterns

### Check if Feature Enabled
```python
aug_cfg = get_augmentation_config(config)
if aug_cfg:
    print("Using augmentation")
else:
    print("No augmentation")
```

### Manual Spectral Crop
```python
loader.apply_spectral_crop(
    crop_region=(700, 1800),
    calib_file='data/RamanFreqTable.csv'
)
```

### Manual Patient Cap
```python
loader.apply_patient_cap(patient_cap=500)
```

### Manual Generator Creation
```python
gen = RamanAugmentationGenerator(
    X=X_train,
    y=y_train,
    batch_size=256,
    max_shift=3,
    mixup_alpha=0.2,
    augment=True,  # or False for inference
    seed=42
)
```

---

## Backward Compatibility

### exp_101 (No Config Keys)
```python
# Works unchanged - no crop_region, patient_cap, augmentation in config
loader = GliomaDataLoader.from_config(config['paths']['processed_file'], config)
# No cropping → full spectrum
# No downsampling → all spectra kept
# No augmentation → standard training
```

### exp_103 (All Config Keys)
```python
# Uses all features - crop_region, patient_cap, augmentation in config
loader = GliomaDataLoader.from_config(config['paths']['processed_file'], config)
# Cropping applied → fingerprint region only
# Downsampling applied → 500/patient/class
# Augmentation available → shifts + mixup
```

---

## Debugging

### Check Loader Applied Preprocessing
```python
print(loader.X.shape)  # Should be cropped size if crop_region in config
```

### Check Generator Output
```python
train_gen = RamanAugmentationGenerator(X_train, y_train, augment=True)
batch_x, batch_y = train_gen[0]
print(batch_x.shape)  # Should be (batch_size, n_features)
```

### Check TATR Data Shape
```python
aug_gen = RamanAugmentationGenerator(X_train, y_train, augment=True)
X_aug = np.vstack([batch_x for batch_x, _ in aug_gen])
X_combined = np.vstack([X_train, X_aug])
print(f"Original: {X_train.shape[0]}")
print(f"Combined: {X_combined.shape[0]}")  # Should be 2x
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| "Calibration file not found" | Check `paths.calib_file` in config, verify file exists |
| Augmentation not working | Check `augmentation.enabled: true`, use `augment=True` in generator |
| Shape mismatch | Ensure crop applied before generator, both same n_features |
| Memory error with TATR | Reduce batch_size or limit augmented samples: `X_aug[:5000]` |
| Test predictions affected by augmentation | Use `augment=False` for test generator |

---

## Files You're Using

| File | What | Import |
|------|------|--------|
| src/data/loader.py | Load data, apply crop/cap | `from src.data.loader import GliomaDataLoader` |
| src/data/augmentation.py | Augmentation + spectral loss | `from src.data.augmentation import RamanAugmentationGenerator, spectral_loss` |
| configs/exp_103.yaml | Configuration | `ConfigManager.load_config('exp_103')` |

---

## Minimal Working Example

```python
# NB02 Classifier with TATR
from src.data.augmentation import get_augmentation_config, RamanAugmentationGenerator
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Setup
config = ConfigManager.load_config('exp_103')
aug_cfg = get_augmentation_config(config)

# TATR
if aug_cfg:
    aug_gen = RamanAugmentationGenerator(X_train, y_train, **aug_cfg, augment=True)
    X_aug = np.vstack([batch_x for batch_x, _ in aug_gen])
    X_train = np.vstack([X_train, X_aug])
    y_train = np.concatenate([y_train, y_train])

# Train
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Test (no augmentation)
preds = rf.predict(X_test)
```

```python
# NB03 CVAE with Augmentation
from src.data.augmentation import RamanAugmentationGenerator, spectral_loss, get_augmentation_config

# Setup
aug_cfg = get_augmentation_config(config)
if aug_cfg:
    train_gen = RamanAugmentationGenerator(X_train, y_train, **aug_cfg, augment=True)
    val_gen = RamanAugmentationGenerator(X_val, y_val, augment=False)
else:
    train_gen, val_gen = None, None

# Compile
cvae.compile(loss=lambda y_t, y_p: spectral_loss(y_t, y_p, 0.7, 0.3))

# Train
cvae.fit(train_gen or X_train, validation_data=(val_gen or X_val), epochs=70)
```

---

## API Summary

### GliomaDataLoader.get_wavenumber_indices()
```python
start_idx, end_idx = loader.get_wavenumber_indices(
    crop_region=(700, 1800),
    calib_file='data/RamanFreqTable.csv'
)
```

### GliomaDataLoader.apply_spectral_crop()
```python
loader.apply_spectral_crop(
    crop_region=(700, 1800),
    calib_file='data/RamanFreqTable.csv'
)
```

### GliomaDataLoader.apply_patient_cap()
```python
loader.apply_patient_cap(patient_cap=500)
```

### RamanAugmentationGenerator.__init__()
```python
gen = RamanAugmentationGenerator(
    X=X_train,
    y=y_train,
    batch_size=256,
    max_shift=3,
    mixup_alpha=0.2,
    augment=True,
    seed=42
)
```

### spectral_loss()
```python
loss = spectral_loss(
    y_true=y_pred,
    y_pred=y_reconstructed,
    mse_weight=0.7,
    cosine_weight=0.3
)
```

### get_augmentation_config()
```python
aug_cfg = get_augmentation_config(config)
# Returns: {'max_shift': 3, 'mixup_alpha': 0.2} or {}
```

---

## See Also

- `EXP_103_IMPLEMENTATION_SUMMARY.md` - Complete overview
- `EXP_103_IMPLEMENTATION_GUIDE.md` - Detailed walkthrough
- `EXP_103_NOTEBOOK_INTEGRATION.md` - Notebook code examples
- `src/data/augmentation.py` - Full API docs in docstrings
