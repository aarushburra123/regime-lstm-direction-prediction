"""
Experiment Logger

Logs hyperparameters, metrics, and training progress to CSV files.
Simple alternative to Weights & Biases or MLflow for tracking experiments.
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    Log experiment hyperparameters and results to CSV.
    
    Creates two files:
    - experiments.csv: One row per experiment with hyperparams and final metrics
    - {experiment_id}_history.csv: Training history for each epoch
    
    Usage:
        logger = ExperimentLogger('experiments/')
        exp_id = logger.start_experiment({'lr': 0.001, 'epochs': 50})
        for epoch in range(epochs):
            logger.log_epoch(epoch, train_loss=0.5, val_loss=0.4)
        logger.end_experiment(test_accuracy=0.55)
    """
    
    def __init__(self, log_dir: str = 'experiments'):
        """
        Args:
            log_dir: Directory to store experiment logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.experiments_file = os.path.join(log_dir, 'experiments.csv')
        self.current_experiment_id = None
        self.current_hyperparams = None
        self.history_file = None
    
    def start_experiment(self, 
                         hyperparams: Dict[str, Any],
                         experiment_name: Optional[str] = None) -> str:
        """
        Start a new experiment.
        
        Args:
            hyperparams: Dictionary of hyperparameters
            experiment_name: Optional name for the experiment
        
        Returns:
            experiment_id: Unique identifier for this experiment
        """
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.current_experiment_id = f"{experiment_name}_{timestamp}"
        else:
            self.current_experiment_id = f"exp_{timestamp}"
        
        self.current_hyperparams = hyperparams.copy()
        self.current_hyperparams['experiment_id'] = self.current_experiment_id
        self.current_hyperparams['start_time'] = datetime.now().isoformat()
        
        # Create history file for this experiment
        self.history_file = os.path.join(
            self.log_dir, 
            f"{self.current_experiment_id}_history.csv"
        )
        
        print(f"Started experiment: {self.current_experiment_id}")
        return self.current_experiment_id
    
    def log_epoch(self, epoch: int, **metrics):
        """
        Log metrics for current epoch.
        
        Args:
            epoch: Current epoch number
            **metrics: Metric name-value pairs (e.g., train_loss=0.5)
        """
        if self.history_file is None:
            raise ValueError("No experiment started. Call start_experiment first.")
        
        row = {'epoch': epoch, **metrics}
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(self.history_file)
        
        with open(self.history_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    
    def end_experiment(self, **final_metrics):
        """
        End current experiment and log final results.
        
        Args:
            **final_metrics: Final metrics (e.g., test_accuracy=0.55)
        """
        if self.current_experiment_id is None:
            raise ValueError("No experiment started.")
        
        # Combine hyperparams and final metrics
        result = self.current_hyperparams.copy()
        result['end_time'] = datetime.now().isoformat()
        result.update(final_metrics)
        
        # Append to experiments CSV
        file_exists = os.path.exists(self.experiments_file)
        
        with open(self.experiments_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)
        
        print(f"Experiment {self.current_experiment_id} completed")
        print(f"Results saved to {self.experiments_file}")
        
        # Reset state
        self.current_experiment_id = None
        self.current_hyperparams = None
        self.history_file = None
    
    def log_config(self, config: Dict[str, Any], filename: str = 'config.json'):
        """Save experiment configuration to JSON."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)


def load_experiment_results(log_dir: str = 'experiments') -> 'pd.DataFrame':
    """
    Load all experiment results as a DataFrame.
    
    Args:
        log_dir: Directory with experiment logs
    
    Returns:
        DataFrame with one row per experiment
    """
    import pandas as pd
    
    experiments_file = os.path.join(log_dir, 'experiments.csv')
    
    if not os.path.exists(experiments_file):
        print(f"No experiments file found at {experiments_file}")
        return pd.DataFrame()
    
    return pd.read_csv(experiments_file)


def load_experiment_history(experiment_id: str, 
                            log_dir: str = 'experiments') -> 'pd.DataFrame':
    """
    Load training history for a specific experiment.
    
    Args:
        experiment_id: ID of the experiment
        log_dir: Directory with experiment logs
    
    Returns:
        DataFrame with training history (one row per epoch)
    """
    import pandas as pd
    
    history_file = os.path.join(log_dir, f"{experiment_id}_history.csv")
    
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"No history file found for {experiment_id}")
    
    return pd.read_csv(history_file)


if __name__ == "__main__":
    print("Testing ExperimentLogger...")
    
    # Create logger
    logger = ExperimentLogger('/tmp/test_experiments')
    
    # Start experiment
    exp_id = logger.start_experiment(
        hyperparams={
            'model': 'SingleLSTM',
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32
        },
        experiment_name='test'
    )
    
    # Log epochs
    for epoch in range(3):
        logger.log_epoch(
            epoch=epoch,
            train_loss=0.5 - epoch*0.1,
            val_loss=0.45 - epoch*0.08,
            train_acc=0.5 + epoch*0.05,
            val_acc=0.52 + epoch*0.04
        )
    
    # End experiment
    logger.end_experiment(
        test_accuracy=0.55,
        test_loss=0.35,
        sharpe_ratio=0.6
    )
    
    # Load and display results
    results = load_experiment_results('/tmp/test_experiments')
    print("\nExperiment results:")
    print(results.to_string())
    
    print("\n✓ ExperimentLogger test passed!")
