import yfinance as yf
import pandas as pd
import numpy as np

def run_backtest(ticker_symbol="AAPL"):
    print(f"📈 Initializing Quant Backtester for {ticker_symbol}...")
    
    # 1. Download Historical Stock Data
    print("🌐 Fetching historical market data from Yahoo Finance...")
    ticker = yf.Ticker(ticker_symbol)
    # Get the last 5 years of daily data
    df = ticker.history(period="5y")
    
    if df.empty:
        print("❌ Failed to download data. Check your internet connection or the ticker symbol.")
        return
        
    print(f"✅ Successfully downloaded {len(df)} days of market data.")
    
    # 2. Build the Moving Average Strategy
    print("🧠 Calculating 20-Day and 50-Day Simple Moving Averages (SMA)...")
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 3. Generate Trading Signals
    # 1 means Buy/Hold, 0 means Sell/Stay in Cash
    df['Signal'] = 0
    # Buy when the fast moving average (20) is higher than the slow moving average (50)
    df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1 
    
    # Calculate daily returns of the stock itself
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Calculate the strategy returns (we only get returns on days we are holding the stock)
    # We shift the signal by 1 day because we trade on yesterday's signal
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    
    # 4. Calculate Performance Metrics
    # Drop empty rows from the moving average calculation
    df = df.dropna()
    
    # Cumulative Returns
    cumulative_market_return = (1 + df['Daily_Return']).cumprod().iloc[-1] - 1
    cumulative_strategy_return = (1 + df['Strategy_Return']).cumprod().iloc[-1] - 1
    
    # Annualized Volatility (Risk)
    trading_days = 252
    strategy_volatility = df['Strategy_Return'].std() * np.sqrt(trading_days)
    
    # Sharpe Ratio (Assuming 0% risk-free rate for simplicity)
    sharpe_ratio = (df['Strategy_Return'].mean() * trading_days) / strategy_volatility

    # 5. Output the Results
    print("\n==========================================")
    print(f"📊 BACKTEST RESULTS: {ticker_symbol} (5 Years)")
    print("==========================================")
    print(f"Buy & Hold Market Return: {cumulative_market_return * 100:.2f}%")
    print(f"Algo Strategy Return:     {cumulative_strategy_return * 100:.2f}%")
    print(f"Strategy Volatility:      {strategy_volatility * 100:.2f}%")
    print(f"Sharpe Ratio:             {sharpe_ratio:.2f}")
    print("==========================================")
    
    # Save the data for future ML training
    export_file = f"{ticker_symbol}_backtest_data.csv"
    df.to_csv(export_file)
    print(f"\n💾 Strategy data saved to '{export_file}' for Neural Network training.")

if __name__ == "__main__":
    # You can change 'AAPL' to 'TSLA', 'MSFT', 'NVDA', or 'BTC-USD'
    run_backtest("AAPL")