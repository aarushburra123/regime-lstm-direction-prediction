"""Tests for regime detection."""
import pandas as pd
from regime_trader.regime.detector import RegimeDetector


def test_vix_regime_detection():
    """Test basic VIX threshold regime detection."""
    detector = RegimeDetector(vix_threshold=20.0)
    
    assert detector.detect_vix_regime(15.0) == 'low_vol'
    assert detector.detect_vix_regime(25.0) == 'high_vol'
    assert detector.detect_vix_regime(20.0) == 'high_vol'


def test_dataframe_labeling():
    """Test regime labeling on DataFrame."""
    detector = RegimeDetector(vix_threshold=20.0)
    
    df = pd.DataFrame({'Close_VIX': [10, 15, 25, 30, 18]})
    result = detector.label_dataframe(df)
    
    assert 'Regime' in result.columns
    assert result.iloc[0]['Regime'] == 'low_vol'
    assert result.iloc[2]['Regime'] == 'high_vol'
