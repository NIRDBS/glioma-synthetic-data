"""
Configuration management for exp_102 pipeline
Loads YAML configs and sets up experiment directories
"""

import yaml
import json
from pathlib import Path


class ConfigManager:
    """Manages experiment configuration from YAML files"""
    
    def __init__(self, config_path):
        """Load configuration from YAML file"""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key, default=None):
        """Get a value from the configuration"""
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def to_dict(self):
        """Return configuration as dictionary"""
        return self.config


def setup_experiment(config_name='exp_102'):
    """
    Setup an experiment by loading its configuration and creating output directories
    
    Args:
        config_name: Name of config file (without .yaml extension)
    
    Returns:
        tuple: (config_dict, output_dir)
    """
    # Find config file
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'configs' / f'{config_name}.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute paths in 'paths' section
    for key in ['raw_data_dir', 'calib_file'] + [k for k in config.get('paths', {}).keys() if 'dir' in k or 'file' in k]:
        if key in config['paths']:
            path_val = config['paths'][key]
            if not Path(path_val).is_absolute():
                config['paths'][key] = str(project_root / path_val)
    
    # Also convert relative paths in 'preprocessing' section (e.g., tumor_maps_path)
    if 'preprocessing' in config:
        for key in config['preprocessing']:
            if isinstance(key, str) and ('path' in key.lower() or 'file' in key.lower() or key.lower().endswith('_dir')):
                val = config['preprocessing'][key]
                if isinstance(val, str) and val and not Path(val).is_absolute():
                    try:
                        config['preprocessing'][key] = str(project_root / val)
                    except (ValueError, TypeError):
                        # If it's not a valid path string, leave it as is
                        pass
    
    # Create output directories
    output_dir = Path(config['paths']['experiment_root'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(config['paths']['data_output_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path(config['paths']['models_baseline_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Return config and output directory
    return config, output_dir


if __name__ == '__main__':
    config, output_dir = setup_experiment('exp_102')
    print(f"Experiment: {config['name']}")
    print(f"Output directory: {output_dir}")
    print(f"Config keys: {list(config.keys())}")
