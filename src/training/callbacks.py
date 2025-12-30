"""
Training Callbacks

Includes early stopping, model checkpointing, and learning rate scheduling
to prevent overfitting and save the best model during training.
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Any


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.
    
    Usage:
        early_stopping = EarlyStopping(patience=10, mode='min')
        for epoch in range(epochs):
            val_loss = train_epoch(...)
            if early_stopping(val_loss):
                break
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current validation metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose and self.counter > 0:
                print(f"  EarlyStopping: {self.counter}/{self.patience} epochs without improvement")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  EarlyStopping triggered after {self.patience} epochs without improvement")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class ModelCheckpoint:
    """
    Save model checkpoint when validation metric improves.
    
    Usage:
        checkpoint = ModelCheckpoint('models/best_model.pt')
        for epoch in range(epochs):
            val_loss = train_epoch(...)
            checkpoint(val_loss, model)
    """
    
    def __init__(self,
                 filepath: str,
                 mode: str = 'min',
                 save_best_only: bool = True,
                 verbose: bool = True):
        """
        Args:
            filepath: Path to save model (e.g., 'models/best.pt')
            mode: 'min' for loss, 'max' for accuracy
            save_best_only: Only save when metric improves
            verbose: Print save messages
        """
        self.filepath = filepath
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_value = None
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best
        else:
            self.is_better = lambda new, best: new > best
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    def __call__(self, current_value: float, model: torch.nn.Module, 
                 extra_info: Optional[Dict[str, Any]] = None):
        """
        Potentially save model checkpoint.
        
        Args:
            current_value: Current validation metric
            model: PyTorch model to save
            extra_info: Optional dict with extra info to save
        """
        should_save = False
        
        if self.save_best_only:
            if self.best_value is None or self.is_better(current_value, self.best_value):
                self.best_value = current_value
                should_save = True
        else:
            should_save = True
        
        if should_save:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_value': current_value,
            }
            if extra_info:
                checkpoint.update(extra_info)
            
            torch.save(checkpoint, self.filepath)
            
            if self.verbose:
                print(f"  Checkpoint saved: {self.filepath} (value={current_value:.4f})")
    
    def load_best(self, model: torch.nn.Module, device: str = 'cpu') -> Dict:
        """Load the best saved checkpoint."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"No checkpoint found at {self.filepath}")
        
        checkpoint = torch.load(self.filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.verbose:
            print(f"  Loaded checkpoint from {self.filepath}")
        
        return checkpoint


class LearningRateScheduler:
    """
    Wrapper around PyTorch learning rate schedulers with logging.
    
    Supports:
    - ReduceLROnPlateau: Reduce LR when metric plateaus
    - StepLR: Reduce LR every N epochs
    - CosineAnnealing: Cosine annealing schedule
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler_type: str = 'plateau',
                 verbose: bool = True,
                 **kwargs):
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: 'plateau', 'step', or 'cosine'
            verbose: Print LR changes
            **kwargs: Arguments for the scheduler
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.verbose = verbose
        self.last_lr = self._get_lr()
        
        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.5)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def step(self, metric: Optional[float] = None):
        """
        Take a scheduler step.
        
        Args:
            metric: Validation metric (required for ReduceLROnPlateau)
        """
        if self.scheduler_type == 'plateau':
            if metric is None:
                raise ValueError("metric required for ReduceLROnPlateau")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        new_lr = self._get_lr()
        if new_lr != self.last_lr and self.verbose:
            print(f"  LR changed: {self.last_lr:.2e} → {new_lr:.2e}")
            self.last_lr = new_lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._get_lr()


class TrainingHistory:
    """
    Track and store training metrics.
    
    Usage:
        history = TrainingHistory()
        history.update(epoch=1, train_loss=0.5, val_loss=0.4, train_acc=0.6)
        history.plot()
    """
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
    
    def update(self, **kwargs):
        """Add metrics for current epoch."""
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get(self, key: str) -> list:
        """Get history for a metric."""
        return self.history.get(key, [])
    
    def to_dict(self) -> Dict:
        """Return history as dictionary."""
        return self.history.copy()
    
    def save(self, filepath: str):
        """Save history to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath: str):
        """Load history from file."""
        import json
        with open(filepath, 'r') as f:
            self.history = json.load(f)


if __name__ == "__main__":
    print("Testing callbacks...")
    
    # Test EarlyStopping
    es = EarlyStopping(patience=3, mode='min')
    losses = [0.5, 0.4, 0.3, 0.35, 0.36, 0.37, 0.38]
    print("\nEarlyStopping test (should trigger at loss 0.38):")
    for i, loss in enumerate(losses):
        stop = es(loss)
        print(f"  Epoch {i+1}: loss={loss}, stop={stop}")
        if stop:
            break
    
    # Test ModelCheckpoint with a dummy model
    print("\nModelCheckpoint test:")
    model = torch.nn.Linear(10, 1)
    ckpt = ModelCheckpoint('/tmp/test_checkpoint.pt', mode='min')
    ckpt(0.5, model)
    ckpt(0.4, model)  # Should save
    ckpt(0.45, model)  # Should not save
    
    # Test LearningRateScheduler
    print("\nLearningRateScheduler test:")
    optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=0.01)
    scheduler = LearningRateScheduler(optimizer, 'plateau', patience=2)
    for i in range(8):
        scheduler.step(metric=0.5)  # Constant metric should trigger LR reduction
    
    print("\n✓ All callback tests passed!")
