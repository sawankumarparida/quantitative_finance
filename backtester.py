import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Phase 1: Data Pipeline ---
ticker = "RELIANCE.NS" 
start_date = "2020-01-01"
end_date = "2024-01-01"

print(f"Downloading data for {ticker}...")
df = yf.download(ticker, start=start_date, end=end_date)

# THE FIX: Create a fresh DataFrame and use .squeeze() to flatten the yfinance output
# into a clean, 1D array that Plotly and Numpy can understand.
data = pd.DataFrame()
data['Close'] = df['Close'].squeeze()
data.dropna(inplace=True)

# --- Phase 2: Mathematical Strategy Engine ---
short_window = 20
long_window = 50

# Calculate Rolling Moving Averages
data['SMA20'] = data['Close'].rolling(window=short_window).mean()
data['SMA50'] = data['Close'].rolling(window=long_window).mean()

# Generate Trading Signals
data['Signal'] = 0.0
data['Signal'] = np.where(data['SMA20'] > data['SMA50'], 1.0, 0.0)

# Isolate the exact crossover moments (Position Changes)
data['Position'] = data['Signal'].diff()

print("\nData tracking with calculated signals:")
print(data.tail(10))

# --- Phase 2: Interactive Plotly Visualization ---
print("\nLaunching interactive graph...")
fig = go.Figure()

# 1. Add the Close Price Line
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                         mode='lines', name='Close Price', 
                         line=dict(color='rgba(0,0,0,0.3)', width=1.5)))

# 2. Add the Fast SMA
fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], 
                         mode='lines', name='20-Day SMA', 
                         line=dict(color='blue', width=2)))

# 3. Add the Slow SMA
fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], 
                         mode='lines', name='50-Day SMA', 
                         line=dict(color='orange', width=2)))

# 4. Add Buy Markers
buys = data[data['Position'] == 1.0]
fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], 
                         mode='markers', name='BUY Signal', 
                         marker=dict(color='green', size=12, symbol='triangle-up')))

# 5. Add Sell Markers
sells = data[data['Position'] == -1.0]
fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], 
                         mode='markers', name='SELL Signal', 
                         marker=dict(color='red', size=12, symbol='triangle-down')))

# 6. Configure Interactive Zoom, Pan, and Range Sliders
fig.update_layout(
    title=f"{ticker} Interactive Strategy Backtester",
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    hovermode="x unified",
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis=dict(
        autorange=True,
        fixedrange=False
    )
)

# Open the interactive graph in your default web browser
fig.show()