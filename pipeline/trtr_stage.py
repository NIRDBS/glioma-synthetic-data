"""
Stage 2: TRTR (Train on Real, Test on Real)
Trains baseline models using stratified group CV splits and saves models
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from src.data.loader import GliomaDataLoader
from src.models.baselines import ApolloPipeline


def trtr_stage(config, output_dir):
    """
    Train baseline models on real data (TRTR).
    
    Args:
        config: Configuration dict
        output_dir: Output directory for model results
    
    Returns:
        Dict with fold predictions and model paths
    """
    
    print(f"\n{'='*80}")
    print(f"TRTR STAGE: Train Baselines on Real Data")
    print(f"{'='*80}")
    
    # Load data
    data_path = Path(config['paths']['processed_file'])
    print(f"\nLoading data from: {data_path}")
    
    loader = GliomaDataLoader(
        data_path=data_path,
        split_file=config['cv_strategy'].get('split_source'),
        memory_mapped=False,
        tumor_only=config['preprocessing'].get('y_tumor', False),
        config=config
    )
    
    X = loader.X
    y = loader.y_class
    groups = loader.groups
    
    print(f"Dataset: {X.shape[0]} spectra, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    print(f"Groups (samples): {len(np.unique(groups))}")
    
    # CV splits
    n_splits = config['cv_strategy'].get('n_folds', 5)
    cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Model output directory
    models_dir = Path(output_dir) / 'models' / 'baselines'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for results
    trtr_results = {
        'fold_preds': [],
        'fold_models': [],
        'fold_indices': [],
        'config': config
    }
    
    print(f"\nTraining Apollo Pipeline across {n_splits} folds...\n")
    
    fold_num = 0
    for train_idx, test_idx in cv_splitter.split(X, y, groups):
        fold_num += 1
        
        print(f"  Fold {fold_num}/{n_splits}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"    Train: {X_train.shape[0]} spectra, Test: {X_test.shape[0]} spectra")
        
        # Train Apollo Pipeline
        apollo = ApolloPipeline(config=config)
        apollo.fit(X_train, y_train)
        
        # Predict
        y_pred = apollo.predict(X_test)
        y_pred_proba = apollo.predict_proba(X_test)
        
        # Save model
        model_path = models_dir / f"apollo_fold_{fold_num-1}.joblib"
        joblib.dump(apollo, model_path)
        print(f"    Saved: {model_path.name}")
        
        # Store results
        trtr_results['fold_preds'].append({
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'test_idx': test_idx
        })
        trtr_results['fold_models'].append(str(model_path))
        trtr_results['fold_indices'].append((train_idx, test_idx))
    
    # Save summary
    summary_path = models_dir / 'trtr_summary.joblib'
    joblib.dump(trtr_results, summary_path)
    
    print(f"\n TRTR training complete")
    print(f"  Models saved to: {models_dir}")
    print(f"  Summary: {summary_path}")
    
    return trtr_results
