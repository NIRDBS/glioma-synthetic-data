"""
Stage 4: TSTR (Train on Synthetic, Test on Real)
Trains baseline models using synthetic data and tests on real data
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from src.data.loader import GliomaDataLoader
from src.models.baselines import ApolloPipeline


def tstr_stage(config, output_dir, cvae_results=None):
    """
    Train baseline models on synthetic data, test on real (TSTR).
    
    Args:
        config: Configuration dict
        output_dir: Output directory for model results
        cvae_results: Output from cvae_stage (contains synthetic_path)
    
    Returns:
        Dict with fold predictions and model paths
    """
    
    print(f"\n{'='*80}")
    print(f"TSTR STAGE: Train Baselines on Synthetic, Test on Real")
    print(f"{'='*80}")
    
    # Load synthetic data
    synthetic_path = Path(cvae_results['synthetic_path'])
    print(f"\nLoading synthetic data from: {synthetic_path}")
    
    synthetic_data = np.load(synthetic_path, allow_pickle=True)
    
    # Prepare synthetic X and y
    X_synthetic_list = []
    y_synthetic_list = []
    
    for key in sorted(synthetic_data.files):
        class_id = int(key.split('_')[-1])
        spectra = synthetic_data[key]
        X_synthetic_list.append(spectra)
        y_synthetic_list.extend([class_id] * len(spectra))
    
    X_synthetic = np.vstack(X_synthetic_list)
    y_synthetic = np.array(y_synthetic_list)
    
    print(f"Synthetic dataset: {X_synthetic.shape[0]} spectra, {X_synthetic.shape[1]} features")
    print(f"Classes: {np.unique(y_synthetic)}")
    
    # Load real test data (for testing across CV folds)
    data_path = Path(config['paths']['processed_file'])
    print(f"\nLoading real data from: {data_path}")
    
    loader = GliomaDataLoader(
        data_path=data_path,
        split_file=config['cv_strategy'].get('split_source'),
        memory_mapped=False,
        tumor_only=config['preprocessing'].get('y_tumor', False),
        config=config
    )
    
    X_real = loader.X
    y_real = loader.y_class
    groups = loader.groups
    
    print(f"Real dataset: {X_real.shape[0]} spectra, {X_real.shape[1]} features")
    
    # CV splits (same as TRTR for fair comparison)
    n_splits = config['cv_strategy'].get('n_folds', 5)
    cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Model output directory
    models_dir = Path(output_dir) / 'models' / 'tstr'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for results
    tstr_results = {
        'fold_preds': [],
        'fold_models': [],
        'fold_indices': [],
        'config': config
    }
    
    print(f"\nTraining Apollo Pipeline on synthetic data across {n_splits} folds...\n")
    
    fold_num = 0
    for train_idx, test_idx in cv_splitter.split(X_real, y_real, groups):
        fold_num += 1
        
        print(f"  Fold {fold_num}/{n_splits}")
        
        # Use ALL synthetic data for training
        X_train = X_synthetic
        y_train = y_synthetic
        
        # Use real data for testing
        X_test = X_real[test_idx]
        y_test = y_real[test_idx]
        
        print(f"    Train (synthetic): {X_train.shape[0]} spectra")
        print(f"    Test (real): {X_test.shape[0]} spectra")
        
        # Train Apollo Pipeline
        apollo = ApolloPipeline(config=config)
        apollo.fit(X_train, y_train)
        
        # Predict on real test data
        y_pred = apollo.predict(X_test)
        y_pred_proba = apollo.predict_proba(X_test)
        
        # Save model
        model_path = models_dir / f"apollo_fold_{fold_num-1}.joblib"
        joblib.dump(apollo, model_path)
        print(f"    Saved: {model_path.name}")
        
        # Store results
        tstr_results['fold_preds'].append({
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'test_idx': test_idx
        })
        tstr_results['fold_models'].append(str(model_path))
        tstr_results['fold_indices'].append((None, test_idx))  # No train_idx for synthetic
    
    # Save summary
    summary_path = models_dir / 'tstr_summary.joblib'
    joblib.dump(tstr_results, summary_path)
    
    print(f"\n TSTR training complete")
    print(f"  Models saved to: {models_dir}")
    print(f"  Summary: {summary_path}")
    
    return tstr_results
