import yfinance as yf
import pandas as pd

print("Testing yfinance downloads...\n")

# Download SPY
spy = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)
print("SPY columns:", spy.columns.tolist())
print("SPY shape:", spy.shape)
print("\nSPY head:")
print(spy.head())

# Download VIX
vix = yf.download('^VIX', start='2020-01-01', end='2024-12-31', progress=False)
print("\n" + "="*60)
print("VIX columns:", vix.columns.tolist())
print("VIX shape:", vix.shape)
print("\nVIX head:")
print(vix.head())

# Test merge
print("\n" + "="*60)
print("Testing merge...")
combined = pd.merge(
    spy[['Close']],
    vix[['Close']],
    left_index=True,
    right_index=True,
    how='inner',
    suffixes=('_SPY', '_VIX')
)
print("Combined columns:", combined.columns.tolist())
print("\nCombined head:")
print(combined.head())
