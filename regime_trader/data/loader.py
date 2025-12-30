"""Data loading and pipeline for SPY/VIX."""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Handles downloading and processing market data."""
    
    def __init__(self, start_date: str = '2020-01-01', 
                 end_date: str = '2024-12-31',
                 data_dir: str = 'data'):
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_spy(self) -> pd.DataFrame:
        """Download SPY price data."""
        spy = yf.download('SPY', start=self.start_date, 
                         end=self.end_date, progress=False)
        
        # Flatten MultiIndex columns if present
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = ['_'.join(col).strip() if isinstance(col, tuple) 
                          else col for col in spy.columns.values]
        
        # Calculate returns
        close_col = [col for col in spy.columns if 'Close' in col][0]
        spy['Returns'] = spy[close_col].pct_change()
        
        return spy
    
    def download_vix(self) -> pd.DataFrame:
        """Download VIX data."""
        vix = yf.download('^VIX', start=self.start_date, 
                         end=self.end_date, progress=False)
        
        # Flatten MultiIndex columns
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = ['_'.join(col).strip() if isinstance(col, tuple) 
                          else col for col in vix.columns.values]
        
        return vix
    
    def run_pipeline(self) -> pd.DataFrame:
        """Execute full data pipeline."""
        print("Downloading data...")
        spy = self.download_spy()
        vix = self.download_vix()
        
        # Merge
        combined = pd.merge(
            spy[['Close_SPY' if 'SPY' in spy.columns[0] else 'Close', 'Returns']],
            vix[['Close_^VIX' if 'VIX' in vix.columns[0] else 'Close']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Rename for clarity
        if 'Close' in combined.columns:
            combined.rename(columns={'Close': 'Close_VIX'}, inplace=True)
        
        # Save
        filepath = self.data_dir / 'spy_vix_combined.csv'
        combined.to_csv(filepath)
        print(f"✓ Saved to {filepath}")
        
        return combined
