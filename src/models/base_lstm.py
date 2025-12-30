"""
Single LSTM Baseline Model

This is a standard 2-layer LSTM trained on ALL data (no regime switching).
Purpose: Prove that LSTM works before building the more complex MoE architecture.

If this model can't beat 50% accuracy, the regime-switching approach won't help either.
Build and verify this FIRST before proceeding to regime_lstm.py.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SingleLSTM(nn.Module):
    """
    Single LSTM model for binary direction prediction.
    
    Architecture:
        Input (seq_len, n_features) 
        → LSTM Layer 1 (hidden_size=64)
        → LSTM Layer 2 (hidden_size=32)
        → Fully Connected → Sigmoid → Prediction (0 or 1)
    
    This is the baseline LSTM to beat with the Mixture-of-Experts approach.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size_1: int = 64,
                 hidden_size_2: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = False):
        """
        Args:
            input_size: Number of input features per timestep
            hidden_size_1: Hidden units in first LSTM layer
            hidden_size_2: Hidden units in second LSTM layer
            num_layers: Number of LSTM layers (stacked)
            dropout: Dropout rate between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(SingleLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0,  # Dropout only between layers
            bidirectional=bidirectional
        )
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1 * self.num_directions,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size_2 * self.num_directions, 1)
        # Note: No Sigmoid here. We output logits for use with BCEWithLogitsLoss
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to help convergence."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Initialize forget gate bias to 1 to remember long-term dependencies
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            Logits of shape (batch_size,)
        """
        batch_size = x.size(0)
        
        # LSTM layer 1
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout_layer(lstm1_out)
        
        # LSTM layer 2
        lstm2_out, (h_n, c_n) = self.lstm2(lstm1_out)
        
        # Use the last hidden state from the final layer
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_final = h_n[-1]
        
        # Apply dropout
        h_final = self.dropout_layer(h_final)
        
        # Fully connected layer
        logit = self.fc(h_final)
        
        return logit.squeeze(-1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Make binary predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            predictions = (probs >= threshold).long()
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x: Input tensor
        
        Returns:
            Probabilities of class 1 (UP direction)
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        return probs


def train_single_lstm(model: SingleLSTM,
                      train_loader,
                      val_loader,
                      epochs: int = 50,
                      learning_rate: float = 1e-3,
                      device: str = 'cpu',
                      class_weight: Optional[float] = None,
                      patience: int = 10,
                      verbose: bool = True) -> Tuple[list, list, list, list]:
    """
    Train the SingleLSTM model.
    
    Args:
        model: SingleLSTM instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Maximum training epochs
        learning_rate: Initial learning rate
        device: 'cpu' or 'cuda'
        class_weight: Positive class weight for imbalanced data
        patience: Early stopping patience
        verbose: Print progress
    
    Returns:
        train_losses, val_losses, train_accs, val_accs
    """
    model = model.to(device)
    
    # Use BCEWithLogitsLoss for numerical stability
    if class_weight is not None:
        pos_weight = torch.tensor([class_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
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
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            logits = model(sequences)
            loss = criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * sequences.size(0)
            
            # Apply sigmoid for accuracy calculation
            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).long()
            train_correct += (predictions == targets.long()).sum().item()
            train_total += targets.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
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
                val_total += len(targets) # Corrected from len(targets).size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"Restored best model with val_loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, train_accs, val_accs


def evaluate_single_lstm(model: SingleLSTM,
                         test_loader,
                         device: str = 'cpu') -> dict:
    """
    Evaluate SingleLSTM on test data.
    
    Args:
        model: Trained SingleLSTM
        test_loader: Test DataLoader
        device: Device for inference
    
    Returns:
        Dictionary with accuracy, predictions, probabilities, actuals
    """
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_probabilities = []
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
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    accuracy = (all_predictions == all_targets).mean()
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'actuals': all_targets
    }


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing SingleLSTM...")
    
    # Create dummy data
    batch_size = 32
    seq_len = 20
    n_features = 10
    
    # Random input
    x = torch.randn(batch_size, seq_len, n_features)
    
    # Create model
    model = SingleLSTM(input_size=n_features)
    print(f"Model architecture:")
    print(model)
    
    # Forward pass
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test predict
    predictions = model.predict(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction distribution: {predictions.sum().item()}/{len(predictions)} = {predictions.float().mean():.2%} UP")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ SingleLSTM test passed!")
