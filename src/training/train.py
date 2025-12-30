"""
Main Training Script for Regime-Switching LSTM

Orchestrates the entire training pipeline:
1. Loads engineered features and pre-fitted scaler
2. Trains SingleLSTM baseline
3. Trains Mixture-of-Experts LSTM (3-phase strategy)
4. Logs experiments and saves checkpoints

Usage:
    python src/training/train.py --epochs 50 --batch_size 32
"""

import os
import argparse
import torch
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from src.data.dataset import (
    load_engineered_features, 
    create_dataloaders, 
    create_regime_specific_dataloaders
)
from src.models.base_lstm import SingleLSTM, train_single_lstm, evaluate_single_lstm
from src.models.regime_lstm import (
    MixtureOfExpertsModel, 
    train_moe_phase1, 
    train_moe_phase2, 
    train_moe_phase3
)
from src.training.logger import ExperimentLogger
from src.training.evaluate import evaluate_by_regime

def parse_args():
    parser = argparse.ArgumentParser(description='Train Regime-Switching LSTM')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models_dir', type=str, default='models', help='Models directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--phases', type=str, default='all', choices=['all', 'single', 'moe'], 
                        help='Training phases to run')
    parser.add_argument('--debug', action='store_true', help='Debug mode (fast run)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup directories
    os.makedirs(args.models_dir, exist_ok=True)
    log_dir = os.path.join(args.models_dir, 'logs')
    logger = ExperimentLogger(log_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("\n" + "="*50)
    print("Loading Data")
    print("="*50)
    
    try:
        df = load_engineered_features(args.data_dir)
    except FileNotFoundError:
        print("ERROR: features_engineered.csv not found. Please run feature engineering notebook first.")
        return
    
    if args.debug:
        print("DEBUG MODE: Using subset of data")
        df = df.iloc[-1000:]  # Last 1000 days only to ensure enough high-vol samples
        args.epochs = 2
        args.seq_len = 5
    
    # Load separate scaler to ensure we don't re-fit
    scaler_path = os.path.join(args.models_dir, 'feature_scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"WARNING: Scaler not found at {scaler_path}. Please run feature engineering.")
    else:
        print(f"Found scaler at {scaler_path}")
    
    # Create DataLoaders
    train_loader, val_loader, test_loader, feature_names = create_dataloaders(
        df, 
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        scaler_path=scaler_path
    )
    
    input_size = len(feature_names)
    print(f"Input features: {input_size}")
    
    # Check class imbalance
    train_targets = train_loader.dataset.targets.numpy()
    up_pct = (train_targets == 1).mean()
    print(f"Class Balance (Train): {up_pct:.1%} UP / {1-up_pct:.1%} DOWN")
    
    class_weight = None
    if up_pct < 0.45 or up_pct > 0.55:
        # Calculate weight for positive class
        # weight = (1 / pos_count) * (total / 2)
        # Simplified: weight = (1 - up_pct) / up_pct
        class_weight = (1 - up_pct) / up_pct
        print(f"Imbalance detected. Using positive class weight: {class_weight:.2f}")
    
    # 2. Train SingleLSTM Baseline
    if args.phases in ['all', 'single']:
        print("\n" + "="*50)
        print("Training SingleLSTM Baseline")
        print("="*50)
        
        single_lstm = SingleLSTM(
            input_size=input_size,
            hidden_size_1=args.hidden_size,
            hidden_size_2=args.hidden_size // 2,
            dropout=args.dropout
        )
        
        exp_id = logger.start_experiment(
            hyperparams={
                'model': 'SingleLSTM',
                **vars(args),
                'class_weight': class_weight
            },
            experiment_name='SingleLSTM'
        )
        
        train_losses, val_losses, train_accs, val_accs = train_single_lstm(
            single_lstm,
            train_loader,
            val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            class_weight=class_weight
        )
        
        # Log history
        for i in range(len(train_losses)):
            logger.log_epoch(
                i, 
                train_loss=train_losses[i], 
                val_loss=val_losses[i],
                train_acc=train_accs[i],
                val_acc=val_accs[i]
            )
        
        # Evaluate on test set
        metrics = evaluate_single_lstm(single_lstm, test_loader, device=device)
        print(f"\nSingleLSTM Test Accuracy: {metrics['accuracy']:.4f}")
        
        logger.end_experiment(test_accuracy=metrics['accuracy'])
        
        # Save model
        torch.save(single_lstm.state_dict(), os.path.join(args.models_dir, 'single_lstm.pt'))
        print("Saved SingleLSTM model.")

    
    # 3. Train Mixture-of-Experts
    if args.phases in ['all', 'moe']:
        print("\n" + "="*50)
        print("Training Mixture-of-Experts Model")
        print("="*50)
        
        moe_model = MixtureOfExpertsModel(
            input_size=input_size,
            expert_hidden_size=args.hidden_size,
            expert_dropout=args.dropout,
            gating_hidden_size=32
        )
        
        exp_id = logger.start_experiment(
            hyperparams={
                'model': 'MoE_LSTM',
                **vars(args)
            },
            experiment_name='MoE_LSTM'
        )
        
        # Get regime-specific loaders for Phase 1
        regime_loaders = create_regime_specific_dataloaders(
            df, 
            sequence_length=args.seq_len,
            batch_size=args.batch_size
        )
        
        # Phase 1: Pre-train experts
        print("\nPhase 1: Pre-training experts...")
        train_moe_phase1(
            moe_model,
            regime_loaders['low_vol'][0],  # Low vol train loader
            regime_loaders['high_vol'][0], # High vol train loader
            epochs=max(5, args.epochs // 2), # Fewer epochs for pre-training
            learning_rate=args.lr,
            device=device
        )
        
        # Phase 2: Train gating network
        print("\nPhase 2: Training gating network...")
        train_moe_phase2(
            moe_model,
            train_loader,
            val_loader,
            epochs=max(5, args.epochs // 2),
            learning_rate=args.lr,
            device=device
        )
        
        # Phase 3: Fine-tune end-to-end
        print("\nPhase 3: Fine-tuning end-to-end...")
        t_loss, v_loss, t_acc, v_acc = train_moe_phase3(
            moe_model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            learning_rate=args.lr / 5, # Lower LR for fine-tuning
            device=device
        )
        
        # Log history (only for phase 3 for simplicity, or aggregate)
        for i in range(len(t_loss)):
            logger.log_epoch(
                i, 
                train_loss=t_loss[i], 
                val_loss=v_loss[i],
                train_acc=t_acc[i],
                val_acc=v_acc[i],
                phase=3
            )
        
        # Evaluate
        print("\nEvaluting MoE Model:")
        regime_metrics = evaluate_by_regime(moe_model, test_loader, device=device)
        
        print(f"  Overall Accuracy:  {regime_metrics['overall']['accuracy']:.4f}")
        print(f"  Low Vol Accuracy:  {regime_metrics['low_vol']['accuracy']:.4f}")
        print(f"  High Vol Accuracy: {regime_metrics['high_vol']['accuracy']:.4f}")
        
        logger.end_experiment(
            test_accuracy=regime_metrics['overall']['accuracy'],
            low_vol_acc=regime_metrics['low_vol']['accuracy'],
            high_vol_acc=regime_metrics['high_vol']['accuracy']
        )
        
        # Save model
        torch.save(moe_model.state_dict(), os.path.join(args.models_dir, 'moe_lstm.pt'))
        print("Saved MoE model.")

if __name__ == "__main__":
    main()
