"""Market regime detection."""
import pandas as pd


class RegimeDetector:
    """Detects market volatility regimes."""
    
    def __init__(self, vix_threshold: float = 20.0):
        self.vix_threshold = vix_threshold
    
    def detect_vix_regime(self, vix_level: float) -> str:
        """Simple VIX threshold-based regime detection."""
        return 'high_vol' if vix_level >= self.vix_threshold else 'low_vol'
    
    def label_dataframe(self, df: pd.DataFrame, 
                        vix_col: str = 'Close_VIX') -> pd.DataFrame:
        """Add regime labels to dataframe."""
        df = df.copy()
        df['Regime'] = df[vix_col].apply(self.detect_vix_regime)
        return df
