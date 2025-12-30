"""
Unified Training Pipeline for Regime-Switching LSTM

Orchestrates the entire workflow:
1. Feature Loading & Validation
2. Baseline Model Evaluation (Random, Momentum, Persistence)
3. SingleLSTM Training & Verification
4. MoE LSTM Training (3-Phase)
5. Model Comparison & Reporting

Addresses critical issues:
- proper class balance checks
- logits/BCEWithLogitsLoss for stability
- direct comparison against baselines
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

# Import local modules
from src.data.dataset import (
    load_engineered_features,
    create_dataloaders,
    create_regime_specific_dataloaders
)
from src.models.baselines import (
    RandomBaseline, 
    MomentumBaseline, 
    NaivePersistenceBaseline,
    LogisticRegressionBaseline
)
from src.models.base_lstm import SingleLSTM, train_single_lstm, evaluate_single_lstm
from src.models.regime_lstm import (
    MixtureOfExpertsModel,
    train_moe_phase1,
    train_moe_phase2,
    train_moe_phase3
)
from src.training.evaluate import evaluate_model, evaluate_by_regime
from src.training.logger import ExperimentLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Train Full Pipeline')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models_dir', type=str, default='models', help='Models directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--debug', action='store_true', help='Debug mode (fast run)')
    return parser.parse_args()

def get_test_dataframe(df, train_ratio=0.7, val_ratio=0.15, forward_buffer=5):
    """Reconstruct test dataframe split for baseline evaluation."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    test_start = val_end + forward_buffer
    
    if test_start >= n:
        return df.iloc[-10:] # Fallback for debug/small data
        
    return df.iloc[test_start:]

def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.models_dir, exist_ok=True)
    log_dir = os.path.join(args.models_dir, 'logs')
    logger = ExperimentLogger(log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load & Validate Features
    print("\n" + "="*50)
    print("1. FEATURE LOADING")
    print("="*50)
    
    try:
        df = load_engineered_features(args.data_dir)
    except FileNotFoundError:
        print("ERROR: features_engineered.csv not found. Running feature engineering...")
        # Use simple os.system or import to run engineering if missing?
        # For now, just fail.
        raise
        
    if args.debug:
        print("DEBUG MODE: Using last 1000 days")
        df = df.iloc[-1000:]
        # Use smaller seq_len in debug to avoid data scarcity errors
        args.seq_len = 5
        args.epochs = 5
    
    # Check data quality
    print(f"Data shape: {df.shape}")
    print(f"Regimes: {df['Regime'].value_counts().to_dict()}")
    
    # Create DataLoaders
    train_loader, val_loader, test_loader, feature_names = create_dataloaders(
        df, 
        sequence_length=args.seq_len, 
        batch_size=args.batch_size,
        scaler_path=os.path.join(args.models_dir, 'feature_scaler.pkl')
    )
    
    input_size = len(feature_names)
    
    # Check Class Balance
    train_y = train_loader.dataset.targets.numpy()
    val_y = val_loader.dataset.targets.numpy()
    test_y = test_loader.dataset.targets.numpy()
    
    print(f"\nClass Balance:")
    print(f"  Train: {train_y.mean():.1%} UP")
    print(f"  Val:   {val_y.mean():.1%} UP")
    print(f"  Test:  {test_y.mean():.1%} UP")
    
    class_weight = None
    up_pct = train_y.mean()
    if up_pct < 0.45 or up_pct > 0.55:
        class_weight = (1 - up_pct) / up_pct
        print(f"  -> Imbalance detected. Using positive class weight: {class_weight:.2f}")

    # 2. Baseline Models
    print("\n" + "="*50)
    print("2. BASELINE EVALUATION")
    print("="*50)
    
    # Get test dataframe aligned with test_y
    # Note: test_loader targets are derived from df, but sequence creation might drop start rows
    # However, BaselineModel.evaluate usually needs aligned X and y.
    # We will use the numpy targets from loader for y, and reconstruct X from df.
    # But for simplicity, baselines.py evaluates on (X, y).
    # X needs to be DataFrame for column access (Momentum uses Returns_SPY).
    
    # Reconstruct X_test corresponding to test_loader
    # The dataset class aligns (X[t-seq:t], y[t]). y[t] is target at time t.
    # MomentumBaseline predicts y[t] using X[t-1]...
    # So we need the DataFrame rows corresponding to the TEST PERIOD.
    # test_loader.dataset.indices gives the original indices? No, it's a Subset?
    # TimeSeriesDataset creates sequences.
    
    # Let's use the helper to get roughly the same test set
    df_test = get_test_dataframe(df, forward_buffer=5)
    
    # Note: df_test length might differ slightly from test_loader length due to seq_len trimming
    # But for baselines, we just evaluate on the dataframe period.
    # The metrics might strictly differ from LSTM test set by a few samples, 
    # but should be comparable.
    
    X_test_df = df_test
    # Need to match simple targets: default target uses Forward_5d_return > 0
    # Recalculate y_test for this DF slice to be sure
    y_test_baseline = X_test_df['Direction_label'].values
    
    baselines = {
        'Random': RandomBaseline(),
        'Momentum': MomentumBaseline(),
        'Persistence': NaivePersistenceBaseline()
    }
    
    results = []
    print(f"Evaluating baselines on {len(X_test_df)} samples...")
    
    for name, model in baselines.items():
        metrics = model.evaluate(X_test_df, y_test_baseline)
        metrics['model'] = name
        print(f"  {name}: Acc={metrics['accuracy']:.4f}, Sharpe={metrics.get('sharpe', 0):.2f}")
        results.append(metrics)
        
    baseline_df = pd.DataFrame(results)
    best_baseline = baseline_df.loc[baseline_df['accuracy'].idxmax()]
    print(f"\nBest Baseline: {best_baseline['model']} ({best_baseline['accuracy']:.4f})")
    target_acc = max(0.52, best_baseline['accuracy'] + 0.01)
    print(f"TARGET ACCURACY: > {target_acc:.4f}")

    # 3. Single LSTM
    print("\n" + "="*50)
    print("3. SINGIE LSTM TRAINING")
    print("="*50)
    
    single_lstm = SingleLSTM(
        input_size=input_size,
        hidden_size_1=args.hidden_size,
        hidden_size_2=args.hidden_size // 2,
        dropout=args.dropout
    )
    
    logger.start_experiment(
        hyperparams={'model': 'SingleLSTM', **vars(args)},
        experiment_name='SingleLSTM'
    )
    
    t_loss, v_loss, t_acc, v_acc = train_single_lstm(
        single_lstm,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        class_weight=class_weight
    )
    
    # Diagnostics: Check probability distribution
    single_lstm.eval()
    with torch.no_grad():
        # Get one batch
        batch = next(iter(val_loader))
        if len(batch) == 3:
            seqs, _, _ = batch
        else:
            seqs, _ = batch
            
        seqs = seqs.to(device)
        logits = single_lstm(seqs)
        probs = torch.sigmoid(logits)
        print(f"\nDiagnostics (Val Batch):")
        print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"  Probs range:  [{probs.min():.3f}, {probs.max():.3f}]")
        print(f"  Pred UP ratio: {(probs > 0.5).long().float().mean():.2%}")
    
    # Test
    metrics = evaluate_single_lstm(single_lstm, test_loader, device=device)
    print(f"\nSingleLSTM Test Accuracy: {metrics['accuracy']:.4f}")
    
    logger.end_experiment(test_accuracy=metrics['accuracy'])
    torch.save(single_lstm.state_dict(), os.path.join(args.models_dir, 'single_lstm.pt'))
    
    if metrics['accuracy'] < 0.50:
        print("\nWARNING: SingleLSTM failed to beat random guess!")
        # proceed anyway to debug MoE
    
    # 4. MoE LSTM
    print("\n" + "="*50)
    print("4. MoE LSTM TRAINING")
    print("="*50)
    
    moe_model = MixtureOfExpertsModel(
        input_size=input_size,
        expert_hidden_size=args.hidden_size,
        expert_dropout=args.dropout
    )
    
    logger.start_experiment(
        hyperparams={'model': 'MoE', **vars(args)},
        experiment_name='MoE'
    )
    
    # Split for experts
    regime_loaders = create_regime_specific_dataloaders(
        df, args.seq_len, args.batch_size
    )
    
    # Phase 1
    print("\n[Phase 1] Expert Pre-training")
    train_moe_phase1(
        moe_model, 
        regime_loaders['low_vol'][0], 
        regime_loaders['high_vol'][0],
        epochs=args.epochs // 2,
        learning_rate=args.lr,
        device=device
    )
    
    # Phase 2
    print("\n[Phase 2] Gating Network")
    train_moe_phase2(
        moe_model, train_loader, val_loader,
        epochs=args.epochs // 2,
        learning_rate=args.lr,
        device=device
    )
    
    # Phase 3
    print("\n[Phase 3] Fine-tuning")
    train_moe_phase3(
        moe_model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.lr / 5,
        device=device
    )
    
    # Eval
    regime_metrics = evaluate_by_regime(moe_model, test_loader, device=device)
    overall_acc = regime_metrics['overall']['accuracy']
    
    print(f"\nMoE Test Results:")
    print(f"  Overall:  {overall_acc:.4f}")
    print(f"  Low Vol:  {regime_metrics['low_vol']['accuracy']:.4f}")
    print(f"  High Vol: {regime_metrics['high_vol']['accuracy']:.4f}")
    
    logger.end_experiment(test_accuracy=overall_acc)
    torch.save(moe_model.state_dict(), os.path.join(args.models_dir, 'moe_lstm.pt'))
    
    # 5. Final Comparison
    print("\n" + "="*50)
    print("FINAL LEADERBOARD")
    print("="*50)
    
    summary = [
        {'Model': name, 'Accuracy': m['accuracy']} for name, m in zip(baselines.keys(), results)
    ]
    summary.append({'Model': 'SingleLSTM', 'Accuracy': metrics['accuracy']})
    summary.append({'Model': 'MoE LSTM', 'Accuracy': overall_acc})
    
    df_summary = pd.DataFrame(summary).sort_values('Accuracy', ascending=False)
    print(df_summary)

if __name__ == "__main__":
    main()
