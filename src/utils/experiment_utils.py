"""Experiment setup shared by the pipeline stage command-line entry points."""

import os
from pathlib import Path
from typing import Any, Dict, Tuple

from src.utils.config_manager import ConfigManager


def setup_experiment(config_name: str, verbose: bool = True) -> Tuple[Dict[str, Any], Path]:
    """Load ``configs/{config_name}.yaml``, move to the project root and create the output directory.

    Returns the loaded configuration and ``results/<output_base>``.
    Raises FileNotFoundError if the config does not exist and ValueError if it is invalid.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("EXPERIMENT SETUP")
        print("=" * 80)

    # Relative paths in the configs are relative to the project root.
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / 'configs').exists() and (parent / 'src').exists():
            os.chdir(parent)
            if verbose:
                print(f"\nWorking directory: {Path.cwd()}")
            break

    try:
        config = ConfigManager.load_config(config_name)
    except FileNotFoundError as e:
        print(f"\n!!! Error: {e}")
        raise
    except ValueError as e:
        print(f"\n!!! Configuration validation error: {e}")
        raise

    output_base = config.get('output', {}).get('output_base', config_name)
    output_dir = Path('results') / output_base
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n Loaded configuration: {config_name}")
        print(f"   Description: {config.get('description', 'N/A')}")
        print(f" Output directory: {output_dir.resolve()}")

        pp_cfg = config.get('preprocessing', {})
        cv_cfg = config.get('cv_strategy', {})
        models_cfg = config.get('models', {})
        print("\n" + "-" * 80)
        print("Configuration summary:")
        print("-" * 80)
        print(f"  Crop region: {pp_cfg.get('crop_region', 'N/A')}")
        print(f"  Split source: {cv_cfg.get('split_source', 'N/A')}")
        if 'random_forest' in models_cfg:
            print(f"  RF n_estimators: {models_cfg['random_forest'].get('n_estimators', 'N/A')}")
        if 'cvae' in models_cfg:
            print(f"  CVAE epochs: {models_cfg['cvae'].get('epochs', 'N/A')}")
        print("\n" + "=" * 80 + "\n")

    return config, output_dir
