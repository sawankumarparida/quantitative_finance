import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define the asset and timeframe
# "RELIANCE.NS" targets Reliance on the National Stock Exchange. 
# You can change this to "^NSEI" for NIFTY 50, or "AAPL" for Apple.
ticker = "RELIANCE.NS" 
start_date = "2020-01-01"
end_date = "2024-01-01"

# 2. Fetch the historical data
print(f"Downloading data for {ticker}...")
df = yf.download(ticker, start=start_date, end=end_date)

# 3. Clean and isolate the data we need
# Yahoo Finance provides Open, High, Low, Close, and Volume. 
# For a simple moving average, we only care about the 'Close' price.
data = df[['Close']].copy()
data.dropna(inplace=True) # Remove any missing/corrupt data points

print("\nFirst 5 rows of our dataset:")
print(data.head())

# 4. Plot the closing price to verify our data pipeline
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label=f"{ticker} Closing Price", color='blue', alpha=0.7)
plt.title(f"Historical Price Data for {ticker}")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()