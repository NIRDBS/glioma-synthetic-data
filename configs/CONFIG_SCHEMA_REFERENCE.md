# Experiment Configuration Schema Reference

This document defines the structure and all available parameters for experiment configuration files.

## YAML Configuration Structure

All experiment configs are YAML files with the following top-level sections:

```yaml
# Experiment Metadata
name: string                          # Unique experiment identifier (e.g., "baseline_exp_001")
description: string                   # Human-readable description
version: string                       # Config version (for backwards compatibility tracking)

# Data Preprocessing Configuration
preprocessing:
  # Signal processing parameters
  airpls_lambda: int                  # Baseline correction strength (default: 100)
  airpls_porder: int                  # Polynomial order for baseline (default: 1)
  
  # Spectral dimensions and cropping
  signal_length: int                  # Total spectral dimension (default: 1738)
  crop_region: [start, end] or null   # Wavenumber range to keep (cm⁻¹), or null for full spectrum
                                      # (default: [700, 1800] for fingerprint region)
                                      # Full spectrum typically ranges ~400-3500 cm⁻¹
  
  # Data filtering
  y_tumor: bool                       # true: filter to tumor-only spectra
                                      # false: use all spectra (no filtering)
                                      # (default: true for backward compatibility)
  ignore_samples: [string]            # Sample IDs to exclude (may be empty)

# Cross-Validation and Data Split Strategy
cv_strategy:
  split_source: string                # Path to splits JSON file relative to project root
                                      # (e.g., "data/splits/strict_splits.json")
  use_group_aware_cv: bool            # Whether to use patient-level grouping (default: true)
  holdout_test_set: bool              # Whether to use separate held-out test set (default: true)
  n_folds: int                        # Number of CV folds (inferred from split_source if not specified)

# Model Hyperparameters
models:
  random_forest:
    n_estimators: int                 # Number of trees (default: 300)
    max_depth: int                    # Maximum tree depth (default: 10)
    criterion: string                 # Split criterion: "gini" or "entropy" (default: "entropy")
    min_samples_leaf: int             # Minimum samples per leaf (default: 50)
    min_samples_split: int            # Minimum samples to split node (default: 50)
    random_state: int                 # Random seed for reproducibility (default: 42)
  
  svm:
    C: float                          # Regularization parameter (default: 1.0)
    kernel: string                    # Kernel type: "linear", "rbf", "poly" (default: "rbf")
    gamma: string or float            # Kernel coefficient (default: "scale")
    probability: bool                 # Whether to compute probabilities (default: true)
    random_state: int                 # Random seed (default: 42)
  
  feature_selection:
    n_features: int                   # Number of top features to select (default: 20)
    method: string                    # Selection method (default: "mutual_info")
  
  cvae:
    batch_size: int                   # Batch size for training (default: 256)
    epochs: int                       # Number of training epochs (default: 45)
    learning_rate: float              # Adam learning rate (default: 0.001)
    latent_dim: int                   # Latent space dimensionality (default: 32)
    beta: float                       # KL divergence weight (default: 2.0)
    random_state: int                 # Random seed (default: 42)
    validation_split: float           # Validation data fraction (default: 0.1)

# Output Configuration
output:
  output_base: string                 # Base directory relative to results/
                                      # (e.g., "exp_001" -> "results/exp_001/")
  h5_filename: string                 # Output H5 filename (default: "glioma_dataset.h5")
  include_experiment_name_in_output: bool  # Suffix output files with experiment name (default: false)
  save_intermediate: bool             # Save intermediate processing steps (default: false)
  verbosity: string                   # Logging level: "debug", "info", "warning" (default: "info")
```

## Example Configuration

```yaml
name: baseline_exp_001
description: "Baseline RF+SVM classification without synthetic data"
version: "1.0"

preprocessing:
  airpls_lambda: 100
  airpls_porder: 1
  signal_length: 1738
  crop_region: [700, 1800]  # Fingerprint region, or use null for full spectrum
  y_tumor: false            # Use all spectra (true = tumor-only)
  ignore_samples: []

cv_strategy:
  split_source: "data/splits/strict_splits.json"
  use_group_aware_cv: true
  holdout_test_set: true

models:
  random_forest:
    n_estimators: 300
    max_depth: 10
    criterion: "entropy"
    min_samples_leaf: 50
    min_samples_split: 50
    random_state: 42
  
  svm:
    C: 1.0
    kernel: "rbf"
    gamma: "scale"
    probability: true
    random_state: 42
  
  feature_selection:
    n_features: 20
    method: "mutual_info"
  
  cvae:
    batch_size: 256
    epochs: 45
    learning_rate: 0.001
    latent_dim: 32
    beta: 2.0
    random_state: 42
    validation_split: 0.1

output:
  output_base: "baseline_exp_001"
  h5_filename: "glioma_dataset.h5"
  include_experiment_name_in_output: false
  save_intermediate: false
  verbosity: "info"
```

## Configuration Loading & Validation

### Required Fields
The following fields are **required** in every config file:
- `name`
- `description`
- `preprocessing.airpls_lambda`
- `preprocessing.signal_length`
- `preprocessing.crop_region`
- `cv_strategy.split_source`
- `output.output_base`

### Optional Fields with Defaults
All other fields have sensible defaults. If not specified in the config, defaults will be used automatically.

### Schema Validation
The `ConfigManager` class validates configs against this schema:
- Type checking (int, float, bool, string, list)
- Required field presence
- Directory/file path existence (split_source file must exist)
- Value constraints (e.g., n_features > 0, C > 0)

### Example Usage

```python
from src.utils.config_manager import ConfigManager

# Load configuration by name
config = ConfigManager.load_config("baseline_exp_001")

# Access nested values
lambda_param = config['preprocessing']['airpls_lambda']
split_source = config['cv_strategy']['split_source']
rf_estimators = config['models']['random_forest']['n_estimators']
```

## Extending the Schema

If you need to add new parameters:
1. Document them in this schema reference
2. Add them to the pydantic validator in `ConfigManager`
3. Update all config files that should use the new parameter
4. Add corresponding code in the modules that read these parameters

## Version Tracking

Configs can include a `version` field to track schema evolution:
- Version 1.0: Initial schema (this version)
- Future versions may add/deprecate fields

This allows the loader to apply migrations if the schema changes.
