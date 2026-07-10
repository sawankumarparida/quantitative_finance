# MUST BE AT THE ABSOLUTE TOP BEFORE ANY OTHER IMPORTS
import matplotlib
matplotlib.use('Agg') # Force Matplotlib to headless mode in WSL to prevent segfaults

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime

# 1. Page Configuration
st.set_page_config(page_title="Quant Finance AI", layout="wide")
st.title("📈 Quantitative Finance & AI Trading Dashboard")
st.markdown("Build algorithmic trading strategies and predict future prices using a GPU-accelerated **PyTorch LSTM Neural Network**.")

# 2. Sidebar for User Input
st.sidebar.header("Configure Strategy")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, NVDA, TSLA, BTC-USD)", "AAPL")
years = st.sidebar.slider("Historical Data (Years)", 1, 10, 5)

# Function to fetch data using exact dates to bypass yfinance period limitations
@st.cache_data
def load_data(ticker, years):
    end_date = datetime.date.today()
    # Calculate the exact start date by subtracting the number of years (365 days * years)
    start_date = end_date - datetime.timedelta(days=365 * years)
    
    # Fetch data using start and end dates instead of the restrictive 'period' string
    df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    df = df.dropna()
    return df

data = load_data(ticker, years)

if data.empty:
    st.error("Failed to load data. Please check the ticker symbol.")
else:
    # 3. Algorithmic Backtesting Section
    st.subheader(f"1. Golden Cross Backtest: {ticker}")
    
    # Calculate Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Generate Signals & Returns
    data['Signal'] = 0
    data.loc[data['SMA_20'] > data['SMA_50'], 'Signal'] = 1
    data['Daily_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']
    
    # Calculate Metrics
    clean_data = data.dropna()
    cum_market = (1 + clean_data['Daily_Return']).cumprod().iloc[-1] - 1
    cum_strategy = (1 + clean_data['Strategy_Return']).cumprod().iloc[-1] - 1
    volatility = clean_data['Strategy_Return'].std() * np.sqrt(252)
    sharpe = (clean_data['Strategy_Return'].mean() * 252) / volatility if volatility > 0 else 0

    # Display Metrics in Columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Market Return", f"{cum_market*100:.2f}%")
    col2.metric("Strategy Return", f"{cum_strategy*100:.2f}%")
    col3.metric("Strategy Volatility", f"{volatility*100:.2f}%")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # Plot the Moving Averages
    st.line_chart(data[['Close', 'SMA_20', 'SMA_50']])

    # 4. Deep Learning Prediction Section
    st.divider()
    st.subheader("2. PyTorch Deep Learning Predictor (LSTM)")
    st.markdown("Train a custom Neural Network to predict price movements based on historical sequences.")
    
    if st.button("🚀 Train AI & Predict"):
        with st.spinner("Initializing PyTorch and compiling CUDA tensors..."):
            
            # Setup Device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Preprocess Data
            close_prices = clean_data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            
            sequence_length = 60
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            train_size = int(len(X) * 0.8)
            X_train = torch.tensor(X[:train_size], dtype=torch.float32).to(device)
            y_train = torch.tensor(y[:train_size], dtype=torch.float32).to(device)
            X_test = torch.tensor(X[train_size:], dtype=torch.float32).to(device)
            y_test = torch.tensor(y[train_size:], dtype=torch.float32).to(device)

            # Define PyTorch Model Architecture inside the function
            class LSTMModel(nn.Module):
                def __init__(self):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(1, 50, 2, batch_first=True)
                    self.linear = nn.Linear(50, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.linear(out[:, -1, :])

            model = LSTMModel().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Train Model
            epochs = 50 # Lowered to 50 for faster web app response
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                preds = model(X_train)
                loss = criterion(preds.squeeze(), y_train)
                loss.backward()
                optimizer.step()
                
                # Update UI
                if (epoch+1) % 5 == 0:
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"Training Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

            # Evaluate Model
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test).cpu().numpy()
                
            test_preds = scaler.inverse_transform(test_preds)
            actuals = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
            
            mae = np.mean(np.abs(test_preds - actuals))
            st.success(f"Training Complete! Mean Absolute Error (MAE): **${mae:.2f}** per share")

            # Plot Results
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(actuals, color='blue', label='Actual Price')
            ax.plot(test_preds, color='red', linestyle='dashed', label='AI Prediction')
            ax.set_title(f"{ticker} Deep Learning Prediction vs Reality")
            ax.legend()
            st.pyplot(fig)