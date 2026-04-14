# EXP 103+ Notebook Integration: Augmentation & Spectral Loss

## Overview
This guide shows the specific code updates for NB02 and NB03 to use exp_103 features:
- Spectral cropping (700-1800 cm⁻¹)
- Patient-level downsampling
- Configurable augmentation (shifts + intra-class mixup)
- CVAE spectral loss (MSE + cosine similarity)

---

## Notebook 02: Classification Baseline (TATR Integration)

### Change 1: Add Augmentation Import (Cell 1)
```python
# EXISTING IMPORTS
from src.utils.config_manager import ConfigManager
from src.notebooks.experiment_utils import setup_experiment
import numpy as np

# ADD THIS LINE
from src.data.augmentation import RamanAugmentationGenerator, get_augmentation_config
```

### Change 2: RF/XGBoost with TATR (After Data Loading)
```python
# EXISTING CODE LOADS: X_train, y_train, X_val, y_val

# NEW: Extract augmentation config
aug_cfg = get_augmentation_config(config)

if aug_cfg:
    print("✓ Augmentation enabled: generating TATR mix...")
    
    # Create generator with augmentation
    aug_gen = RamanAugmentationGenerator(
        X_train, y_train,
        batch_size=32,
        **aug_cfg,
        augment=True
    )
    
    # Collect all augmented samples (one epoch)
    X_aug_list = []
    for batch_x, _ in aug_gen:
        X_aug_list.append(batch_x)
    
    X_aug = np.vstack(X_aug_list)
    
    # Combine real + augmented
    X_train_tatr = np.vstack([X_train, X_aug])
    y_train_tatr = np.concatenate([y_train, y_train])
    
    print(f"  Real: {X_train.shape[0]} → Augmented: {X_aug.shape[0]}")
    print(f"  Combined: {X_train_tatr.shape[0]}")
else:
    X_train_tatr = X_train
    y_train_tatr = y_train
    print("⚠️  Augmentation disabled: using real data only")

# Train on TATR data (real + augmented)
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_tatr, y_train_tatr)  # Train on combined

xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=6)
xgb_model.fit(X_train_tatr, y_train_tatr)  # Train on combined
```

### Change 3: 1D-CNN with Augmentation (Separate Cell)
```python
# Use generator directly for neural network (batched training)

aug_cfg = get_augmentation_config(config)

if aug_cfg:
    # Training generator WITH augmentation
    train_gen = RamanAugmentationGenerator(
        X_train, y_train,
        batch_size=32,
        **aug_cfg,
        augment=True
    )
    
    # Validation generator WITHOUT augmentation
    val_gen = RamanAugmentationGenerator(
        X_val, y_val,
        batch_size=32,
        augment=False  # ← Important: no aug on validation
    )
    
    cnn.fit(train_gen, validation_data=val_gen, epochs=50)
else:
    # Fallback for old configs
    X_train_3d = X_train.reshape(-1, X_train.shape[1], 1)
    X_val_3d = X_val.reshape(-1, X_val.shape[1], 1)
    cnn.fit(X_train_3d, y_train, validation_data=(X_val_3d, y_val), epochs=50)
```

---

## Notebook 03: Synthetic CVAE (Augmented Pre-training)

### Change 1: Add Imports (Cell 1)
```python
# EXISTING
from src.utils.config_manager import ConfigManager
from src.notebooks.experiment_utils import setup_experiment

# ADD THESE
from src.data.augmentation import (
    RamanAugmentationGenerator, 
    spectral_loss, 
    get_augmentation_config
)
```

### Change 2: Create Augmentation Generators (After Data Loading)
```python
# EXISTING: X_train, y_train, X_val, y_val loaded via loader

# NEW: Set up augmentation
aug_cfg = get_augmentation_config(config)

print(f"Augmentation enabled: {bool(aug_cfg)}")

if aug_cfg:
    print(f"Config: max_shift={aug_cfg.get('max_shift')}, alpha={aug_cfg.get('mixup_alpha')}")
    
    # Training generator: AUGMENT
    train_gen = RamanAugmentationGenerator(
        X_train, y_train,
        batch_size=config['models']['cvae']['batch_size'],
        **aug_cfg,
        augment=True
    )
    
    # Validation generator: NO AUGMENT
    val_gen = RamanAugmentationGenerator(
        X_val, y_val,
        batch_size=config['models']['cvae']['batch_size'],
        augment=False
    )
else:
    train_gen = None
    val_gen = None
```

### Change 3: Compile CVAE with Spectral Loss (New Cell)
```python
# EXISTING: cvae model built

loss_type = config['models']['cvae'].get('loss_function', 'mse_cosine_combo')

print(f"Loss function: {loss_type}")

if loss_type == 'mse_cosine_combo':
    mse_weight = config['models']['cvae'].get('gamma', 0.7)
    cos_weight = config['models']['cvae'].get('delta', 0.3)
    
    print(f"  MSE weight: {mse_weight}")
    print(f"  Cosine weight: {cos_weight}")
    
    def custom_loss(y_true, y_pred):
        return spectral_loss(y_true, y_pred, 
                           mse_weight=mse_weight, 
                           cosine_weight=cos_weight)
    
    cvae.compile(
        optimizer=Adam(learning_rate=config['models']['cvae']['learning_rate']),
        loss=custom_loss
    )
else:
    # Fallback: standard MSE
    cvae.compile(
        optimizer=Adam(learning_rate=config['models']['cvae']['learning_rate']),
        loss='mse'
    )
```

### Change 4: Train CVAE (Replace fit() call)
```python
print("\nTraining CVAE...")

if train_gen is not None:
    # Train with augmented data generators
    history = cvae.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['models']['cvae']['epochs'],
        verbose=1
    )
else:
    # Fallback for old configs (no augmentation)
    history = cvae.fit(
        X_train, X_train,
        validation_split=config['models']['cvae']['validation_split'],
        batch_size=config['models']['cvae']['batch_size'],
        epochs=config['models']['cvae']['epochs'],
        verbose=1
    )

print("✓ Training complete")
```

---

## exp_103.yaml Config

Minimal example showing all new keys:

```yaml
name: exp_103
description: "TSAR: Fingerprint crop (700-1800), patient cap, augmentation"

paths:
  calib_file: data/RamanFreqTable.csv  # For wavenumber → index mapping

preprocessing:
  # Spectral cropping
  crop_region: [700, 1800]             # Raman shifts in cm⁻¹ (NEW)
  patient_cap: 500                     # Max spectra per patient/class (NEW)
  y_tumor: true

# NEW SECTION: Augmentation
augmentation:
  enabled: true
  max_shift: 3                         # ±3 pixel wavenumber shift
  mixup_alpha: 0.2                     # Beta distribution param

models:
  cvae:
    batch_size: 256
    epochs: 70
    learning_rate: 0.001
    loss_function: "mse_cosine_combo"  # NEW: combined loss
    gamma: 0.7                         # MSE weight (NEW)
    delta: 0.3                         # Cosine weight (NEW)
```

---

## Key Implementation Points

### 1. Augmentation is Training-Only
```python
# ✓ CORRECT: augment=True for training
train_gen = RamanAugmentationGenerator(X_train, y_train, augment=True)

# ✓ CORRECT: augment=False for validation/test
val_gen = RamanAugmentationGenerator(X_val, y_val, augment=False)
test_gen = RamanAugmentationGenerator(X_test, y_test, augment=False)

# ✗ WRONG: Don't augment inference
# cnn.predict(X_test_augmented)  # Will break test results
```

### 2. TATR Data Combination
```python
# Generate augmented, THEN combine with real
X_aug = np.vstack([batch_x for batch_x, _ in aug_gen])  # Augmented
X_combined = np.vstack([X_train, X_aug])                # Real + Augmented
y_combined = np.concatenate([y_train, y_train])         # Labels repeat

# ✓ Train on combined
classifier.fit(X_combined, y_combined)
```

### 3. Intra-Class Mixup Safety
- Only blends spectra from **same tumor type**
- Different classes: λ=1.0 (no blending)
- Labels are **not** mixed (stay pure)

### 4. Patient Cap Application
- Applied per (patient_id, class) group
- Groups with < cap: kept all
- Groups with > cap: random sample to cap
- Result: balanced, no memorization

### 5. Spectral Cropping from YAML
- User specifies cm⁻¹: `crop_region: [700, 1800]`
- Loader reads RamanFreqTable.csv
- Finds closest indices: e.g., [215, 590]
- Slices: X[:, 215:590]

---

## Testing Backward Compatibility

### exp_101 (Old - No Changes)
```python
# Should work without any notebook changes
config = ConfigManager.load_config('exp_101')
loader = GliomaDataLoader.from_config(config['paths']['processed_file'], config)

# No crop_region → full spectrum used
# No patient_cap → all spectra kept
# No augmentation → standard fit()
```

### exp_103 (New - Full Features)
```python
# Uses all new features automatically
config = ConfigManager.load_config('exp_103')
loader = GliomaDataLoader.from_config(config['paths']['processed_file'], config)

# crop_region applied → 700-1800 cm⁻¹
# patient_cap applied → 500/patient/class
# augmentation enabled → shifts + mixup in generators
```

---

## Integration Checklist

- [ ] Add augmentation imports to NB02 and NB03
- [ ] Wrap TATR code in `if aug_cfg:` block
- [ ] Pass generators to CVAE.fit() and 1D-CNN.fit()
- [ ] Set custom spectral_loss in cvae.compile()
- [ ] exp_103.yaml has all new keys (crop_region, patient_cap, augmentation, loss_function, gamma, delta)
- [ ] Test exp_101 notebook (should work unchanged)
- [ ] Test exp_103 notebook (should use crop, cap, augment, loss)
- [ ] Verify no augmentation on validation/test sets
- [ ] Check patient_cap downsampling in logs

---

## Copy-Paste Code Blocks

### Complete TATR Block (RF/XGBoost)
```python
from src.data.augmentation import get_augmentation_config, RamanAugmentationGenerator
import numpy as np

# Get augmentation config
aug_cfg = get_augmentation_config(config)

# Apply TATR if augmentation enabled
if aug_cfg:
    aug_gen = RamanAugmentationGenerator(X_train, y_train, batch_size=32, **aug_cfg, augment=True)
    X_aug = np.vstack([batch_x for batch_x, _ in aug_gen])
    X_train = np.vstack([X_train, X_aug])
    y_train = np.concatenate([y_train, y_train])

# Train on TATR data
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
```

### Complete CVAE Block with Generators
```python
from src.data.augmentation import spectral_loss, get_augmentation_config, RamanAugmentationGenerator

# Setup
aug_cfg = get_augmentation_config(config)
if aug_cfg:
    train_gen = RamanAugmentationGenerator(X_train, y_train, 
                                          batch_size=256, **aug_cfg, augment=True)
    val_gen = RamanAugmentationGenerator(X_val, y_val, batch_size=256, augment=False)
else:
    train_gen, val_gen = None, None

# Compile with spectral loss
cvae.compile(optimizer='adam',
             loss=lambda y_t, y_p: spectral_loss(y_t, y_p, mse_weight=0.7, cosine_weight=0.3))

# Train
cvae.fit(train_gen or X_train, 
         validation_data=(val_gen or X_val), 
         epochs=70)
```

---

## Notes

- **Config-driven**: Same code works for exp_101/102/103 by changing config name
- **Graceful fallback**: Old configs without new keys work normally
- **No breaking changes**: exp_101 notebooks unchanged, exp_103 just adds optional features
- **Patient-safe**: Intra-class mixup + patient cap maintain research integrity
