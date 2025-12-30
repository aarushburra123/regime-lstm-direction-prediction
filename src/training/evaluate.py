"""
Model Evaluation Utilities

Includes metrics calculation, per-regime evaluation, and model comparison.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def evaluate_model(model: torch.nn.Module,
                   test_loader,
                   device: str = 'cpu',
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model on test data with full metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        device: Device for inference
        threshold: Classification threshold
    
    Returns:
        Dictionary with accuracy, precision, recall, f1, auc
    """
    model = model.to(device)
    model.eval()
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                sequences, targets, _ = batch
            else:
                sequences, targets = batch
            
            sequences = sequences.to(device)
            logits = model(sequences)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = (all_probs >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
    }
    
    # AUC requires varying probabilities
    if len(np.unique(all_probs)) > 1:
        try:
            metrics['auc'] = roc_auc_score(all_targets, all_probs)
        except ValueError:
            metrics['auc'] = 0.5
    else:
        metrics['auc'] = 0.5
    
    return metrics


def evaluate_by_regime(model: torch.nn.Module,
                       test_loader,
                       device: str = 'cpu',
                       threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance separately for each regime.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader (must return (seq, target, regime))
        device: Device for inference
        threshold: Classification threshold
    
    Returns:
        Dictionary with 'low_vol', 'high_vol', and 'overall' metrics
    """
    model = model.to(device)
    model.eval()
    
    all_probs = []
    all_targets = []
    all_regimes = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) != 3:
                raise ValueError("Test loader must return (sequences, targets, regimes)")
            
            sequences, targets, regimes = batch
            sequences = sequences.to(device)
            logits = model(sequences)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_regimes.extend(regimes.numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_regimes = np.array(all_regimes)
    all_preds = (all_probs >= threshold).astype(int)
    
    results = {}
    
    # Overall metrics
    results['overall'] = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'n_samples': len(all_targets)
    }
    
    # Low vol regime (regime=0)
    low_vol_mask = all_regimes == 0
    if low_vol_mask.sum() > 0:
        results['low_vol'] = {
            'accuracy': accuracy_score(all_targets[low_vol_mask], all_preds[low_vol_mask]),
            'precision': precision_score(all_targets[low_vol_mask], all_preds[low_vol_mask], zero_division=0),
            'recall': recall_score(all_targets[low_vol_mask], all_preds[low_vol_mask], zero_division=0),
            'f1': f1_score(all_targets[low_vol_mask], all_preds[low_vol_mask], zero_division=0),
            'n_samples': int(low_vol_mask.sum())
        }
    else:
        results['low_vol'] = {'accuracy': 0.0, 'n_samples': 0}
    
    # High vol regime (regime=1)
    high_vol_mask = all_regimes == 1
    if high_vol_mask.sum() > 0:
        results['high_vol'] = {
            'accuracy': accuracy_score(all_targets[high_vol_mask], all_preds[high_vol_mask]),
            'precision': precision_score(all_targets[high_vol_mask], all_preds[high_vol_mask], zero_division=0),
            'recall': recall_score(all_targets[high_vol_mask], all_preds[high_vol_mask], zero_division=0),
            'f1': f1_score(all_targets[high_vol_mask], all_preds[high_vol_mask], zero_division=0),
            'n_samples': int(high_vol_mask.sum())
        }
    else:
        results['high_vol'] = {'accuracy': 0.0, 'n_samples': 0}
    
    return results


def get_confusion_matrix(model: torch.nn.Module,
                         test_loader,
                         device: str = 'cpu',
                         threshold: float = 0.5) -> np.ndarray:
    """
    Get confusion matrix for model predictions.
    
    Returns:
        2x2 confusion matrix where:
        [[TN, FP],
         [FN, TP]]
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                sequences, targets, _ = batch
            else:
                sequences, targets = batch
            
            sequences = sequences.to(device)
            probs = model(sequences)
            preds = (probs >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    return confusion_matrix(all_targets, all_preds)


def print_classification_report(model: torch.nn.Module,
                                test_loader,
                                device: str = 'cpu',
                                threshold: float = 0.5,
                                target_names: List[str] = None):
    """
    Print detailed classification report.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device
        threshold: Classification threshold
        target_names: Names for classes (default: ['DOWN', 'UP'])
    """
    if target_names is None:
        target_names = ['DOWN', 'UP']
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                sequences, targets, _ = batch
            else:
                sequences, targets = batch
            
            sequences = sequences.to(device)
            probs = model(sequences)
            preds = (probs >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    print(classification_report(all_targets, all_preds, target_names=target_names))


def compare_models(models_dict: Dict[str, torch.nn.Module],
                   test_loader,
                   device: str = 'cpu') -> pd.DataFrame:
    """
    Compare multiple models on the same test set.
    
    Args:
        models_dict: Dictionary mapping model names to models
        test_loader: Test DataLoader
        device: Device for inference
    
    Returns:
        DataFrame with metrics for each model, sorted by accuracy
    """
    results = []
    
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_model(model, test_loader, device)
        metrics['model'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[['model', 'accuracy', 'precision', 'recall', 'f1', 'auc']]
    df = df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    
    return df


def get_predictions_dataframe(model: torch.nn.Module,
                              test_loader,
                              df_test: pd.DataFrame,
                              device: str = 'cpu') -> pd.DataFrame:
    """
    Get predictions as a DataFrame for backtesting.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        df_test: Original test DataFrame with dates
        device: Device for inference
    
    Returns:
        DataFrame with columns: date, actual, predicted, probability
    """
    model = model.to(device)
    model.eval()
    
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                sequences, targets, _ = batch
            else:
                sequences, targets = batch
            
            sequences = sequences.to(device)
            probs = model(sequences)
            preds = (probs >= 0.5).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Create DataFrame
    # Note: Number of predictions may differ from df_test due to sequence creation
    n_predictions = len(all_probs)
    
    # Use the last n_predictions dates from df_test
    if len(df_test) >= n_predictions:
        dates = df_test.index[-n_predictions:].tolist()
    else:
        dates = list(range(n_predictions))
    
    results = pd.DataFrame({
        'date': dates,
        'actual': all_targets,
        'predicted': all_preds,
        'probability': all_probs
    })
    
    results['correct'] = results['actual'] == results['predicted']
    
    return results


if __name__ == "__main__":
    print("Testing evaluation utilities...")
    
    # Create a simple dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(5, 1)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self, x):
            # Take mean of sequence, then predict
            x_mean = x.mean(dim=1)
            return self.sigmoid(self.fc(x_mean)).squeeze(-1)
    
    model = DummyModel()
    
    # Create dummy data
    from torch.utils.data import TensorDataset, DataLoader
    
    n_samples = 50
    seq_len = 10
    n_features = 5
    
    X = torch.randn(n_samples, seq_len, n_features)
    y = torch.randint(0, 2, (n_samples,)).float()
    regimes = torch.randint(0, 2, (n_samples,))
    
    dataset = TensorDataset(X, y, regimes)
    loader = DataLoader(dataset, batch_size=16)
    
    # Test evaluate_model
    print("\nTesting evaluate_model:")
    metrics = evaluate_model(model, loader)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    # Test evaluate_by_regime
    print("\nTesting evaluate_by_regime:")
    regime_results = evaluate_by_regime(model, loader)
    print(f"  Overall: {regime_results['overall']['accuracy']:.4f}")
    print(f"  Low Vol: {regime_results['low_vol']['accuracy']:.4f} ({regime_results['low_vol']['n_samples']} samples)")
    print(f"  High Vol: {regime_results['high_vol']['accuracy']:.4f} ({regime_results['high_vol']['n_samples']} samples)")
    
    # Test confusion matrix
    print("\nConfusion matrix:")
    cm = get_confusion_matrix(model, loader)
    print(cm)
    
    # Test classification report
    print("\nClassification report:")
    print_classification_report(model, loader)
    
    print("\n✓ All evaluation tests passed!")
