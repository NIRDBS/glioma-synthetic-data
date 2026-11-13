"""
Experiment utilities.

Provides setup and utility functions called at the start of each pipeline stage to
ensure consistent configuration and output management.
"""

from pathlib import Path
from src.utils.config_manager import ConfigManager, get_experiment_config
from src.utils.experiment_registry import ExperimentRegistry, register_experiment
from typing import Tuple, Dict, Any
import os


def setup_experiment(config_name: str, verbose: bool = True) -> Tuple[Dict[str, Any], Path]:
    """
    Initialize an experiment by loading configuration and creating output directories.
    
    Call this function once at the start of each stage to set up the experiment.
    This ensures all stages in a pipeline use the same configuration and output the same way.
    
    Args:
        config_name: Name of the experiment (e.g., 'exp_608')
                    Looks for configs/{config_name}.yaml
        verbose: Whether to print setup information
    
    Returns:
        Tuple of (config_dict, output_dir_path) where:
        - config_dict: ConfigManager instance with experiment configuration
        - output_dir_path: Path object for experiment output directory
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    
    Example:
        from src.utils.experiment_utils import setup_experiment
        
        config_name = 'exp_608'  # <-- User edits this line
        config, output_dir = setup_experiment(config_name)
        
        # Now use throughout the stage:
        preprocessing_config = config['preprocessing']
        model_config = config['models']
        
        # Save outputs to:
        output_file = output_dir / 'results.csv'
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("EXPERIMENT SETUP")
        print("=" * 80)
    
    # 0. Find and change to project root (important for relative paths)
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / 'configs').exists() and (parent / 'src').exists():
            os.chdir(parent)
            if verbose:
                print(f"\nWorking directory: {Path.cwd()}")
            break
    
    # 1. Load configuration
    try:
        config = ConfigManager.load_config(config_name)
        if verbose:
            print(f"\n Loaded configuration: {config_name}")
            print(f"   Description: {config.get('description', 'N/A')}")
    except FileNotFoundError as e:
        print(f"\n!!! Error: {e}")
        raise
    except ValueError as e:
        print(f"\n!!! Configuration validation error: {e}")
        raise
    
    # 2. Create output directory
    output_base = config.get('output', {}).get('output_base', config_name)
    output_dir = Path('results') / output_base
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n Output directory: {output_dir.resolve()}")
    
    # 3. Register experiment in registry
    try:
        registry = ExperimentRegistry()
        registry.register_experiment(
            experiment_name=config_name,
            description=config.get('description', 'No description'),
            notes=f"Experiment initialization on {_get_timestamp()}"
        )
        if verbose:
            print(f" Registered in experiment registry")
    except ValueError as e:
        # Experiment already registered, just update run status
        if "already exists" in str(e):
            if verbose:
                print(f"  Experiment already registered (re-running)")
        else:
            raise
    
    # 4. Update registry status to indicate this stage is running
    try:
        registry.update_run_status(
            experiment_name=config_name,
            status='running',
            stage='pipeline'
        )
    except:
        pass  # Registry update is not critical
    
    if verbose:
        print("\n" + "-" * 80)
        print("Configuration Summary:")
        print("-" * 80)
        print(f"  Preprocessing:")
        pp_cfg = config.get('preprocessing', {})
        print(f"    - AirPLS lambda: {pp_cfg.get('airpls_lambda', 'N/A')}")
        print(f"    - Signal length: {pp_cfg.get('signal_length', 'N/A')}")
        print(f"    - Crop region: {pp_cfg.get('crop_region', 'N/A')}")
        
        print(f"\n  CV Strategy:")
        cv_cfg = config.get('cv_strategy', {})
        print(f"    - Split source: {cv_cfg.get('split_source', 'N/A')}")
        print(f"    - Use group-aware CV: {cv_cfg.get('use_group_aware_cv', 'N/A')}")
        print(f"    - Hold out test set: {cv_cfg.get('holdout_test_set', 'N/A')}")
        
        print(f"\n  Models:")
        models_cfg = config.get('models', {})
        if 'random_forest' in models_cfg:
            print(f"    - RF n_estimators: {models_cfg['random_forest'].get('n_estimators', 'N/A')}")
        if 'svm' in models_cfg:
            print(f"    - SVM C: {models_cfg['svm'].get('C', 'N/A')}")
        if 'cvae' in models_cfg:
            print(f"    - CVAE epochs: {models_cfg['cvae'].get('epochs', 'N/A')}")
        
        print("\n" + "=" * 80 + "\n")
    
    return config, output_dir


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()


def _get_timestamp() -> str:
    """Internal helper for timestamp."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_experiment_info(output_dir: Path, config: Dict[str, Any], stage: str = "completion") -> None:
    """
    Save experiment configuration and metadata to output directory for reproducibility.
    
    Creates a {stage}_metadata.json file documenting the exact config used.
    
    Args:
        output_ dir: Output directory path
        config: ConfigManager instance or dict with configuration
        stage: Pipeline stage (preprocessing, training, evaluation, etc.)
    """
    import json
    
    # Extract dict from ConfigManager if needed
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    metadata_file = output_dir / f"{stage}_metadata.json"
    
    metadata = {
        "stage": stage,
        "timestamp": _get_timestamp(),
        "config": config_dict,
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def log_stage_completion(config_name: str, stage: str, success: bool = True) -> None:
    """
    Log completion of a pipeline stage to the experiment registry.
    
    Args:
        config_name: Name of the experiment
        stage: Name of the stage (preprocessing, training, evaluation, etc.)
        success: Whether the stage completed successfully
    """
    try:
        registry = ExperimentRegistry()
        status = "success" if success else "failed"
        registry.update_run_status(config_name, status, stage)
    except Exception as e:
        print(f"!  Could not update registry: {e}")
