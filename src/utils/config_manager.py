"""
Configuration Manager for Experiment Settings

Loads and validates YAML configuration files for glioma-synthetic-data experiments.
Ensures consistent parameters across all stages of a pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class ConfigManager:
    """
    Manages loading, validation, and access to experiment configuration files.
    
    Configurations are YAML files stored in the configs/ directory.
    Each experiment has a single config file that drives all pipeline components.
    
    Example:
        config = ConfigManager.load_config("exp_608")
        crop_region = config['preprocessing']['crop_region']
    """
    
    # Default values for optional configuration parameters
    DEFAULTS = {
        'preprocessing': {
            'signal_length': 1738,
            'crop_region': [700, 1800],
        },
        'cv_strategy': {
            'use_group_aware_cv': True,
            'holdout_test_set': True,
        },
        'models': {
            'random_forest': {
                'n_estimators': 300,
                'max_depth': 10,
                'criterion': 'entropy',
                'min_samples_leaf': 50,
                'min_samples_split': 50,
                'random_state': 42,
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42,
            },
            'feature_selection': {
                'n_features': 20,
                'method': 'mutual_info',
            },
            'cvae': {
                'batch_size': 256,
                'epochs': 45,
                'learning_rate': 0.001,
                'latent_dim': 32,
                'beta': 2.0,
                'random_state': 42,
                'validation_split': 0.1,
            },
        },
        'output': {
            'include_experiment_name_in_output': False,
            'h5_filename': 'glioma_dataset.h5',
            'save_intermediate': False,
            'verbosity': 'info',
        },
    }
    
    # Required fields that must be present in every config
    REQUIRED_FIELDS = [
        'name',
        'description',
        'preprocessing.signal_length',
        'preprocessing.crop_region',
        'cv_strategy.split_source',
        'output.output_base',
    ]
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize ConfigManager with a loaded configuration dictionary."""
        self.config = config_dict
        self._project_root = self._find_project_root()
    
    @staticmethod
    def _find_project_root() -> Path:
        """Find the project root directory (contains configs/ folder)."""
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / 'configs').exists():
                return parent
        # Fallback to current working directory
        return current
    
    @classmethod
    def load_config(cls, experiment_name: str) -> 'ConfigManager':
        """
        Load a configuration by experiment name.
        
        Looks for configs/{experiment_name}.yaml in the project root.
        
        Args:
            experiment_name: Name of the experiment (without .yaml extension)
        
        Returns:
            ConfigManager instance with loaded and validated configuration
        
        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If config fails validation
        """
        project_root = cls._find_project_root()
        config_path = project_root / 'configs' / f'{experiment_name}.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Available configs should be in: {project_root / 'configs'}/"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        if raw_config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")
        
        # Validate the configuration
        cls._validate_config(raw_config, config_path)
        
        # Merge with defaults for optional fields
        merged_config = cls._merge_with_defaults(raw_config)
        
        return cls(merged_config)
    
    @classmethod
    def _validate_config(cls, config: Dict[str, Any], config_path: Path) -> None:
        """
        Validate configuration structure and required fields.
        
        Args:
            config: Configuration dictionary to validate
            config_path: Path to config file (for error messages)
        
        Raises:
            ValueError: If config is invalid
        """
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
        
        # Check required fields
        # Use sentinel value to distinguish between missing and null fields
        MISSING_SENTINEL = object()
        missing_fields = []
        
        for field_path in cls.REQUIRED_FIELDS:
            value = cls._get_nested(config, field_path)
            # Check if field was not found (returns None when missing)
            # Note: crop_region can be None (null), which is valid
            if value is None:
                # Verify the field actually exists in config
                parts = field_path.split('.')
                current = config
                found = True
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        found = False
                        break
                
                # Only report as missing if the field key doesn't exist
                if not found:
                    missing_fields.append(field_path)
        
        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields in {config_path}:\n"
                f"  - " + "\n  - ".join(missing_fields)
            )
        
        # Validate split_source path exists
        split_source = cls._get_nested(config, 'cv_strategy.split_source')
        project_root = cls._find_project_root()
        split_path = project_root / split_source
        
        if not split_path.exists():
            raise ValueError(
                f"CV split file does not exist: {split_path}\n"
                f"Specified in config: {split_source}"
            )
        
        # Type validation for critical fields
        if not isinstance(config.get('name'), str):
            raise ValueError("'name' must be a string")
        
        if not isinstance(config.get('description'), str):
            raise ValueError("'description' must be a string")
        
        preprocessing = config.get('preprocessing', {})
        crop_region = preprocessing.get('crop_region')
        
        # crop_region can be: list [start, end], or null (no cropping)
        if crop_region is not None:
            if not isinstance(crop_region, list):
                raise ValueError("'preprocessing.crop_region' must be a list [start, end] or null (no cropping)")
            
            if len(crop_region) != 2:
                raise ValueError("'preprocessing.crop_region' must have exactly 2 elements or be null")
    
    @classmethod
    def _merge_with_defaults(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge provided config with defaults for missing values.
        
        Args:
            config: User-provided configuration
        
        Returns:
            Merged configuration with defaults filled in
        """
        merged = {}
        
        # Copy all user-provided values
        for key, value in config.items():
            if isinstance(value, dict) and key in cls.DEFAULTS and isinstance(cls.DEFAULTS[key], dict):
                # Recursively merge nested dicts
                merged[key] = cls._merge_dicts(cls.DEFAULTS[key], value)
            else:
                merged[key] = value
        
        # Add missing top-level sections from defaults
        for key, default_value in cls.DEFAULTS.items():
            if key not in merged:
                merged[key] = default_value
        
        return merged
    
    @staticmethod
    def _merge_dicts(defaults: Dict, provided: Dict) -> Dict:
        """Merge provided dict into defaults dict recursively."""
        result = defaults.copy()
        for key, value in provided.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = ConfigManager._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def _get_nested(config: Dict, path: str) -> Any:
        """
        Get a value from nested dictionary using dot notation.
        
        Example:
            _get_nested(config, 'preprocessing.crop_region')
            -> config['preprocessing']['crop_region']
        
        Args:
            config: Dictionary to search
            path: Dot-separated path to value
        
        Returns:
            Value at path, or None if not found
        """
        parts = path.split('.')
        current = config
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access to configuration values."""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for checking if key exists in config."""
        return key in self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback."""
        return self.config.get(key, default)
    
    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Example:
            config.get_nested('models.random_forest.n_estimators', default=300)
        
        Args:
            path: Dot-separated path to value
            default: Default value if path not found
        
        Returns:
            Value at path, or default if not found
        """
        value = self._get_nested(self.config, path)
        return value if value is not None else default
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        name = self.config.get('name', 'Unknown')
        return f"ConfigManager(experiment='{name}')"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"Experiment: {self.config.get('name')}"]
        lines.append(f"Description: {self.config.get('description')}")
        return "\n".join(lines)


def get_experiment_config(experiment_name: str) -> ConfigManager:
    """
    Convenience function to load configuration by experiment name.
    
    This is the primary entry point for loading configs.
    
    Args:
        experiment_name: Name of the experiment (without .yaml extension)
    
    Returns:
        ConfigManager instance with loaded configuration
    
    Example:
        config = get_experiment_config('exp_608')
        rf_params = config['models']['random_forest']
    """
    return ConfigManager.load_config(experiment_name)
