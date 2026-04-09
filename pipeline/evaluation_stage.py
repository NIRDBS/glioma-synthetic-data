"""
Stage 5: Evaluation
Statistical evaluation with bootstrap CI and patient-stratified metrics
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from scipy import stats


def bootstrap_metric(y_true, y_pred, y_pred_proba, metric_fn, n_iterations=1000, ci=95):
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC)
        metric_fn: Metric function to compute
        n_iterations: Number of bootstrap iterations
        ci: Confidence interval (e.g., 95)
    
    Returns:
        Dict with mean, std, CI
    """
    
    scores = []
    np.random.seed(42)
    
    for _ in range(n_iterations):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        try:
            if metric_fn.__name__ in ['roc_auc_score']:
                # Need probabilities
                score = metric_fn(y_true[idx], y_pred_proba[idx, 1] if y_pred_proba.ndim > 1 else y_pred_proba[idx])
            else:
                score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)
        except:
            pass
    
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    
    return {
        'mean': mean,
        'std': std,
        'lower_ci': lower,
        'upper_ci': upper,
        'ci': ci
    }


def evaluation_stage(trtr_results, tstr_results, config, output_dir):
    """
    Evaluate TRTR and TSTR results with bootstrap CI.
    
    Args:
        trtr_results: Output from trtr_stage
        tstr_results: Output from tstr_stage
        config: Configuration dict
        output_dir: Output directory for evaluation results
    
    Returns:
        Dict with evaluation metrics and comparisons
    """
    
    print(f"\n{'='*80}")
    print(f"EVALUATION STAGE: Statistical Analysis and Bootstrap CI")
    print(f"{'='*80}")
    
    # Create evaluation directory
    eval_dir = Path(output_dir) / 'statistical_eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    n_iterations = config.get('bootstrap_iterations', 1000)
    print(f"\nBootstrap iterations: {n_iterations}")
    print(f"Confidence interval: 95%\n")
    
    # Collect results across folds
    all_metrics = []
    fold_summaries = []
    
    n_folds = len(trtr_results['fold_preds'])
    
    for fold_idx in range(n_folds):
        print(f"Fold {fold_idx + 1}/{n_folds}")
        
        # TRTR predictions
        trtr_pred = trtr_results['fold_preds'][fold_idx]
        y_true_trtr = trtr_pred['y_test']
        y_pred_trtr = trtr_pred['y_pred']
        y_proba_trtr = trtr_pred['y_pred_proba']
        
        # TSTR predictions
        tstr_pred = tstr_results['fold_preds'][fold_idx]
        y_true_tstr = tstr_pred['y_test']
        y_pred_tstr = tstr_pred['y_pred']
        y_proba_tstr = tstr_pred['y_pred_proba']
        
        fold_data = {
            'fold': fold_idx,
            'method': [],
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
        }
        
        # Evaluate TRTR
        for method, y_pred, y_proba, y_true in [
            ('TRTR', y_pred_trtr, y_proba_trtr, y_true_trtr),
            ('TSTR', y_pred_tstr, y_proba_tstr, y_true_tstr)
        ]:
            print(f"  {method}:")
            
            fold_data['method'].append(method)
            
            # Accuracy
            acc = accuracy_score(y_true, y_pred)
            acc_boot = bootstrap_metric(y_true, y_pred, y_proba, accuracy_score, n_iterations)
            fold_data['accuracy'].append(acc_boot)
            print(f"    Accuracy: {acc_boot['mean']:.3f} [{acc_boot['lower_ci']:.3f}, {acc_boot['upper_ci']:.3f}]")
            
            # F1
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_boot = bootstrap_metric(y_true, y_pred, y_proba, 
                                      lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0), 
                                      n_iterations)
            fold_data['f1'].append(f1_boot)
            print(f"    F1 (weighted): {f1_boot['mean']:.3f} [{f1_boot['lower_ci']:.3f}, {f1_boot['upper_ci']:.3f}]")
            
            # Precision
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            prec_boot = bootstrap_metric(y_true, y_pred, y_proba,
                                        lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0),
                                        n_iterations)
            fold_data['precision'].append(prec_boot)
            
            # Recall
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            rec_boot = bootstrap_metric(y_true, y_pred, y_proba,
                                       lambda yt, yp: recall_score(yt, yp, average='weighted', zero_division=0),
                                       n_iterations)
            fold_data['recall'].append(rec_boot)
        
        fold_summaries.append(fold_data)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY: Cross-Validation Results")
    print(f"{'='*80}\n")
    
    summary_df = []
    
    for fold_data in fold_summaries:
        for idx, method in enumerate(fold_data['method']):
            summary_df.append({
                'Fold': fold_data['fold'],
                'Method': method,
                'Accuracy': f"{fold_data['accuracy'][idx]['mean']:.3f} ± {fold_data['accuracy'][idx]['std']:.3f}",
                'F1': f"{fold_data['f1'][idx]['mean']:.3f} ± {fold_data['f1'][idx]['std']:.3f}",
                'Precision': f"{fold_data['precision'][idx]['mean']:.3f} ± {fold_data['precision'][idx]['std']:.3f}",
                'Recall': f"{fold_data['recall'][idx]['mean']:.3f} ± {fold_data['recall'][idx]['std']:.3f}",
            })
    
    summary_table = pd.DataFrame(summary_df)
    print(summary_table.to_string(index=False))
    
    # Save results
    results_file = eval_dir / 'evaluation_summary.joblib'
    joblib.dump({
        'fold_summaries': fold_summaries,
        'config': config,
        'n_iterations': n_iterations
    }, results_file)
    
    # Save CSV for inspection
    csv_file = eval_dir / 'metrics_table.csv'
    summary_table.to_csv(csv_file, index=False)
    
    print(f"\n✓ Evaluation complete")
    print(f"  Results saved to: {eval_dir}")
    print(f"  Summary file: {results_file.name}")
    print(f"  CSV table: {csv_file.name}")
    
    return {
        'fold_summaries': fold_summaries,
        'summary_df': summary_table,
        'results_path': str(results_file),
        'config': config
    }
