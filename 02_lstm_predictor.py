import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import os

def run_lstm_predictor(ticker="AAPL"):
    print(f"🤖 Initializing Deep Learning AI for {ticker}...\n")
    
    # Check for CUDA (GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Compute Device: {device}")

    # ==========================================
    # 1. DATA LOADING & PREPROCESSING
    # ==========================================
    csv_file = f"{ticker}_backtest_data.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Error: Could not find {csv_file}. Run the backtester first!")
        return

    print("📊 Loading historical data and scaling...")
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df = df.sort_values('Date')
    
    # We will try to predict the 'Close' price
    close_prices = df['Close'].values.reshape(-1, 1)
    
    # Neural Networks work best with data scaled between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences: Use the past 60 days to predict the 61st day
    sequence_length = 60
    X = []
    y = []
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # Reshape for PyTorch LSTM: (Batch Size, Sequence Length, Features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into 80% Training and 20% Testing
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to PyTorch Tensors and send to GPU
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # ==========================================
    # 2. NEURAL NETWORK ARCHITECTURE
    # ==========================================
    class StockPredictorLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2):
            super(StockPredictorLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM Layer
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            # Output Layer
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # Initialize hidden state and cell state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            
            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))
            # Decode the hidden state of the last time step
            out = self.linear(out[:, -1, :])
            return out

    model = StockPredictorLSTM().to(device)
    criterion = nn.MSELoss() # Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 3. TRAINING LOOP (GPU ACCELERATED)
    # ==========================================
    print(f"\n🧠 Training LSTM Neural Network on {len(X_train)} sequences...")
    epochs = 100
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_train)
        loss = criterion(predictions.squeeze(), y_train)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    print(f"⚡ Training completed in {time.time() - start_time:.2f} seconds!")

    # ==========================================
    # 4. EVALUATION & PREDICTION
    # ==========================================
    print("\n🔮 Predicting unseen test data...")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test).cpu().numpy()
        
    # Un-scale the data back to normal dollar amounts
    test_predictions = scaler.inverse_transform(test_predictions)
    actual_prices = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

    # Calculate average error in dollars
    mae = np.mean(np.abs(test_predictions - actual_prices))
    print(f"🎯 Mean Absolute Error (MAE): ${mae:.2f} per share")

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    print("📈 Generating visualization chart...")
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color='blue', label=f'Actual {ticker} Price')
    plt.plot(test_predictions, color='red', linestyle='dashed', label=f'AI Predicted {ticker} Price')
    plt.title(f'{ticker} Stock Price Prediction (LSTM Neural Network)')
    plt.xlabel('Time (Days into the Test Set)')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.grid(True)
    
    chart_filename = f"{ticker}_AI_Prediction_Chart.png"
    plt.savefig(chart_filename)
    print(f"✅ Success! Chart saved as '{chart_filename}'")
    plt.close()

if __name__ == "__main__":
    # You will need to pip install scikit-learn to run this script!
    run_lstm_predictor("AAPL")