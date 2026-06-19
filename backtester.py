import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Phase 1: Data Pipeline ---
ticker = "RELIANCE.NS" 
start_date = "2020-01-01"
end_date = "2024-01-01"

print(f"Downloading data for {ticker}...")
df = yf.download(ticker, start=start_date, end=end_date)
data = df[['Close']].copy()
data.dropna(inplace=True)

# --- Phase 2: Mathematical Strategy Engine ---
short_window = 20
long_window = 50

# 1. Calculate Rolling Moving Averages
# .rolling() tells Python to look back at a specific window of rows
data['SMA20'] = data['Close'].rolling(window=short_window).mean()
data['SMA50'] = data['Close'].rolling(window=long_window).mean()

# 2. Generate Trading Signals (1 = Hold Asset, 0 = Cash/Out of Market)
# np.where operates like an IF statement: IF SMA20 > SMA50, set to 1, ELSE 0
data['Signal'] = 0.0
data['Signal'] = np.where(data['SMA20'] > data['SMA50'], 1.0, 0.0)

# 3. Isolate the exact crossover moments (Position Changes)
# .diff() subtracts the previous day's signal from today's signal
# Result: +1.0 is a Buy trigger, -1.0 is a Sell trigger, 0.0 means no change
data['Position'] = data['Signal'].diff()

print("\nData tracking with calculated signals:")
print(data.tail(10))

# --- Phase 2 Visualization ---
plt.figure(figsize=(14, 7))

# Plot the asset closing price and the two mathematical lines
plt.plot(data['Close'], label='Close Price', color='black', alpha=0.3)
plt.plot(data['SMA20'], label='20-Day SMA (Fast)', color='blue', alpha=0.75)
plt.plot(data['SMA50'], label='50-Day SMA (Slow)', color='orange', alpha=0.75)

# Plot Buy Triggers (Green Up Arrows)
plt.scatter(data[data['Position'] == 1.0].index, 
            data['SMA20'][data['Position'] == 1.0], 
            label='BUY Signal', marker='^', color='green', s=100, zorder=5)

# Plot Sell Triggers (Red Down Arrows)
plt.scatter(data[data['Position'] == -1.0].index, 
            data['SMA20'][data['Position'] == -1.0], 
            label='SELL Signal', marker='v', color='red', s=100, zorder=5)

plt.title(f"{ticker} - Dual Moving Average Crossover Strategy Triggers")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()