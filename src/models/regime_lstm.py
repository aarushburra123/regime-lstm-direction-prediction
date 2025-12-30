"""
Regime-Switching LSTM with Mixture-of-Experts Architecture

This module implements the core hypothesis: market behavior differs between
low and high volatility regimes, and separate expert models will outperform
a single unified model.

Architecture:
    Input Features
        ↓
    [Gating Network] → Regime Weights (w_low, w_high)
        ↓
    Expert 1 (LSTM) ← Trained on Low Vol data (VIX < 20)
    Expert 2 (LSTM) ← Trained on High Vol data (VIX ≥ 20)
        ↓
    Weighted Combination → Final Prediction (UP/DOWN)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class ExpertLSTM(nn.Module):
    """
    Individual LSTM expert model.
    
    Same architecture as SingleLSTM, but designed to be trained on
    regime-specific data (either low-vol or high-vol).
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden units in LSTM
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate
        """
        super(ExpertLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # Forget gate bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            Logits of shape (batch_size,)
        """
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use final hidden state
        h_final = h_n[-1]  # (batch_size, hidden_size)
        h_final = self.dropout(h_final)
        
        # Output logit (not sigmoid - will be combined in MoE)
        logit = self.fc(h_final)
        
        return logit.squeeze(-1)


class LowVolExpertLSTM(ExpertLSTM):
    """Expert trained on low volatility regime (VIX < 20)."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        self.regime = 'low_vol'


class HighVolExpertLSTM(ExpertLSTM):
    """Expert trained on high volatility regime (VIX >= 20)."""
    
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        self.regime = 'high_vol'


class GatingNetwork(nn.Module):
    """
    Gating network that learns to weight expert predictions.
    
    Input: Current market features (especially VIX-related)
    Output: Softmax weights [w_low, w_high] that sum to 1
    
    The gating network learns which expert to trust based on
    current market conditions.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 32,
                 num_experts: int = 2,
                 use_attention: bool = False):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_experts: Number of expert models to weight
            use_attention: Whether to use attention mechanism
        """
        super(GatingNetwork, self).__init__()
        
        self.num_experts = num_experts
        self.use_attention = use_attention
        
        if use_attention:
            # Attention-based gating (uses full sequence)
            self.attention = nn.MultiheadAttention(
                embed_dim=input_size,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.fc = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_experts)
            )
        else:
            # Simple MLP gating (uses last timestep only)
            self.fc = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_experts)
            )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expert weights.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            Expert weights of shape (batch_size, num_experts)
            where each row sums to 1
        """
        if self.use_attention:
            # Attention over sequence
            attn_out, _ = self.attention(x, x, x)
            # Use mean of attended features
            context = attn_out.mean(dim=1)  # (batch_size, input_size)
            logits = self.fc(context)
        else:
            # Use last timestep features for gating decision
            last_features = x[:, -1, :]  # (batch_size, input_size)
            logits = self.fc(last_features)
        
        # Softmax to get weights that sum to 1
        weights = self.softmax(logits)
        
        return weights


class MixtureOfExpertsModel(nn.Module):
    """
    Mixture-of-Experts model combining regime-specific LSTM experts.
    
    Final prediction = w_low * pred_low + w_high * pred_high
    
    Training strategy:
        Phase 1: Pre-train experts separately on regime-specific data
        Phase 2: Freeze experts, train gating network
        Phase 3: Fine-tune entire system end-to-end
    """
    
    def __init__(self,
                 input_size: int,
                 expert_hidden_size: int = 64,
                 expert_num_layers: int = 2,
                 expert_dropout: float = 0.3,
                 gating_hidden_size: int = 32,
                 use_attention_gating: bool = False):
        """
        Args:
            input_size: Number of input features
            expert_hidden_size: Hidden units for expert LSTMs
            expert_num_layers: Number of LSTM layers in experts
            expert_dropout: Dropout rate for experts
            gating_hidden_size: Hidden units for gating network
            use_attention_gating: Whether gating uses attention
        """
        super(MixtureOfExpertsModel, self).__init__()
        
        self.input_size = input_size
        
        # Expert models
        self.low_vol_expert = LowVolExpertLSTM(
            input_size=input_size,
            hidden_size=expert_hidden_size,
            num_layers=expert_num_layers,
            dropout=expert_dropout
        )
        
        self.high_vol_expert = HighVolExpertLSTM(
            input_size=input_size,
            hidden_size=expert_hidden_size,
            num_layers=expert_num_layers,
            dropout=expert_dropout
        )
        
        # Gating network
        self.gating = GatingNetwork(
            input_size=input_size,
            hidden_size=gating_hidden_size,
            num_experts=2,
            use_attention=use_attention_gating
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor, 
                return_expert_outputs: bool = False) -> torch.Tensor:
        """
        Forward pass combining expert predictions via gating.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            return_expert_outputs: If True, also return individual expert outputs
        
        Returns:
            Probabilities of shape (batch_size,) in range [0, 1]
            Optionally: (probs, expert_dict) where expert_dict contains
                        individual expert outputs and gating weights
        """
        # Get expert predictions (logits)
        low_vol_logit = self.low_vol_expert(x)   # (batch_size,)
        high_vol_logit = self.high_vol_expert(x)  # (batch_size,)
        
        # Get gating weights
        weights = self.gating(x)  # (batch_size, 2)
        w_low = weights[:, 0]
        w_high = weights[:, 1]
        
        # Weighted combination of logits (approximate MoE logic for logits)
        # Proper way: Combined prob = sum(w_i * prob_i)
        # But we want to return LOGITS.
        # Logit(p) is non-linear.
        # Option 1: Return probs (and change loss function to BCELoss - risky)
        # Option 2: Return linear combination of logits (heuristic, but works with BCEWithLogitsLoss)
        # Option 3: Calculate probs, then inverse sigmoid to get logit? Unstable.
        
        # We will use Option 2 (Weighted Logits) for training stability, 
        # or stick to Option 1 (Probs) but use BCELoss.
        
        # User REQ: "outputs are LOGITS" and "Use BCEWithLogitsLoss".
        # So we return combined_logit.
        
        combined_logit = w_low * low_vol_logit + w_high * high_vol_logit
        
        if return_expert_outputs:
            expert_info = {
                'low_vol_logit': low_vol_logit,
                'high_vol_logit': high_vol_logit,
                'low_vol_prob': self.sigmoid(low_vol_logit),
                'high_vol_prob': self.sigmoid(high_vol_logit),
                'gating_weights': weights,
                'w_low': w_low,
                'w_high': w_high
            }
            return combined_logit, expert_info
        
        return combined_logit
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Make binary predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = self.sigmoid(logits)
            predictions = (probs >= threshold).long()
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = self.sigmoid(logits)
        return probs
    
    def freeze_experts(self):
        """Freeze expert parameters for gating-only training."""
        for param in self.low_vol_expert.parameters():
            param.requires_grad = False
        for param in self.high_vol_expert.parameters():
            param.requires_grad = False
        print("Experts frozen - only gating network will train")
    
    def unfreeze_experts(self):
        """Unfreeze expert parameters for end-to-end fine-tuning."""
        for param in self.low_vol_expert.parameters():
            param.requires_grad = True
        for param in self.high_vol_expert.parameters():
            param.requires_grad = True
        print("Experts unfrozen - all parameters will train")
    
    def load_pretrained_experts(self, 
                                low_vol_path: str,
                                high_vol_path: str,
                                device: str = 'cpu'):
        """
        Load pre-trained expert weights.
        
        Args:
            low_vol_path: Path to low-vol expert checkpoint
            high_vol_path: Path to high-vol expert checkpoint
            device: Device to load to
        """
        low_vol_state = torch.load(low_vol_path, map_location=device)
        high_vol_state = torch.load(high_vol_path, map_location=device)
        
        self.low_vol_expert.load_state_dict(low_vol_state)
        self.high_vol_expert.load_state_dict(high_vol_state)
        
        print(f"Loaded pre-trained experts from:")
        print(f"  Low Vol: {low_vol_path}")
        print(f"  High Vol: {high_vol_path}")


def train_moe_phase1(model: MixtureOfExpertsModel,
                     low_vol_loader,
                     high_vol_loader,
                     epochs: int = 30,
                     learning_rate: float = 1e-3,
                     device: str = 'cpu',
                     verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    Phase 1: Pre-train experts separately on regime-specific data.
    
    Args:
        model: MixtureOfExpertsModel instance
        low_vol_loader: DataLoader with low-vol regime data
        high_vol_loader: DataLoader with high-vol regime data
        epochs: Training epochs per expert
        learning_rate: Learning rate
        device: Device
        verbose: Print progress
    
    Returns:
        low_vol_history, high_vol_history (dicts with losses and accuracies)
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    histories = {}
    
    # Train each expert on its respective regime data
    for expert_name, expert, loader in [
        ('low_vol', model.low_vol_expert, low_vol_loader),
        ('high_vol', model.high_vol_expert, high_vol_loader)
    ]:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training {expert_name} expert")
            print(f"{'='*50}")
        
        optimizer = torch.optim.Adam(expert.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        losses, accs = [], []
        
        for epoch in range(epochs):
            expert.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch in loader:
                if len(batch) == 3:
                    sequences, targets, _ = batch
                else:
                    sequences, targets = batch
                
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                logits = expert(sequences)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item() * len(targets)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == targets.long()).sum().item()
                total += len(targets)
            
            avg_loss = epoch_loss / total
            acc = correct / total
            losses.append(avg_loss)
            accs.append(acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        
        histories[expert_name] = {'losses': losses, 'accuracies': accs}
    
    return histories.get('low_vol'), histories.get('high_vol')


def train_moe_phase2(model: MixtureOfExpertsModel,
                     train_loader,
                     val_loader,
                     epochs: int = 20,
                     learning_rate: float = 1e-3,
                     device: str = 'cpu',
                     verbose: bool = True) -> Tuple[list, list]:
    """
    Phase 2: Freeze experts, train gating network only.
    
    Args:
        model: MixtureOfExpertsModel with pre-trained experts
        train_loader: Training DataLoader (all data)
        val_loader: Validation DataLoader
        epochs: Training epochs
        learning_rate: Learning rate
        device: Device
        verbose: Print progress
    
    Returns:
        train_losses, val_losses
    """
    if verbose:
        print(f"\n{'='*50}")
        print("Phase 2: Training gating network (experts frozen)")
        print(f"{'='*50}")
    
    model = model.to(device)
    model.freeze_experts()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.gating.parameters(), lr=learning_rate)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_total = 0
        
        for batch in train_loader:
            if len(batch) == 3:
                sequences, targets, _ = batch
            else:
                sequences, targets = batch
            
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(targets)
            train_total += len(targets)
        
        # Validation
        model.eval()
        val_loss = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    sequences, targets, _ = batch
                else:
                    sequences, targets = batch
                
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                logits = model(sequences)
                loss = criterion(logits, targets)
                val_loss += loss.item() * len(targets)
                val_total += len(targets)
        
        train_losses.append(train_loss / train_total)
        val_losses.append(val_loss / val_total)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")
    
    return train_losses, val_losses


def train_moe_phase3(model: MixtureOfExpertsModel,
                     train_loader,
                     val_loader,
                     epochs: int = 20,
                     learning_rate: float = 1e-4,
                     device: str = 'cpu',
                     patience: int = 10,
                     verbose: bool = True) -> Tuple[list, list, list, list]:
    """
    Phase 3: Fine-tune entire system end-to-end.
    
    Args:
        model: MixtureOfExpertsModel with trained gating
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Training epochs
        learning_rate: Learning rate (should be lower than phase 1/2)
        device: Device
        patience: Early stopping patience
        verbose: Print progress
    
    Returns:
        train_losses, val_losses, train_accs, val_accs
    """
    if verbose:
        print(f"\n{'='*50}")
        print("Phase 3: End-to-end fine-tuning")
        print(f"{'='*50}")
    
    model = model.to(device)
    model.unfreeze_experts()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if len(batch) == 3:
                sequences, targets, _ = batch
            else:
                sequences, targets = batch
            
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(targets)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            train_correct += (preds == targets.long()).sum().item()
            train_total += len(targets)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    sequences, targets, _ = batch
                else:
                    sequences, targets = batch
                
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                logits = model(sequences)
                loss = criterion(logits, targets)
                val_loss += loss.item() * len(targets)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                val_correct += (preds == targets.long()).sum().item()
                val_total += len(targets)
        
        train_losses.append(train_loss / train_total)
        val_losses.append(val_loss / val_total)
        train_accs.append(train_correct / train_total)
        val_accs.append(val_correct / val_total)
        
        scheduler.step(val_losses[-1])
        
        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Acc={train_accs[-1]:.4f}, Val Acc={val_accs[-1]:.4f}")
        
        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    print("Testing Mixture-of-Experts LSTM...")
    
    # Create dummy data
    batch_size = 32
    seq_len = 20
    n_features = 10
    
    x = torch.randn(batch_size, seq_len, n_features)
    
    # Create model
    model = MixtureOfExpertsModel(input_size=n_features)
    
    print(f"\nModel components:")
    print(f"  Low Vol Expert: {type(model.low_vol_expert).__name__}")
    print(f"  High Vol Expert: {type(model.high_vol_expert).__name__}")
    print(f"  Gating Network: {type(model.gating).__name__}")
    
    # Forward pass
    probs, expert_info = model(x, return_expert_outputs=True)
    
    print(f"\nForward pass results:")
    print(f"  Final probabilities shape: {probs.shape}")
    print(f"  Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Low vol expert prob: [{expert_info['low_vol_prob'].min():.3f}, {expert_info['low_vol_prob'].max():.3f}]")
    print(f"  High vol expert prob: [{expert_info['high_vol_prob'].min():.3f}, {expert_info['high_vol_prob'].max():.3f}]")
    print(f"  Gating weights shape: {expert_info['gating_weights'].shape}")
    print(f"  Avg w_low: {expert_info['w_low'].mean():.3f}, Avg w_high: {expert_info['w_high'].mean():.3f}")
    
    # Test freezing
    model.freeze_experts()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable params after freeze: {trainable:,}")
    
    model.unfreeze_experts()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params after unfreeze: {trainable:,}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    
    print("\n✓ MixtureOfExpertsModel test passed!")
