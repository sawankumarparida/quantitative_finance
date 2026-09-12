import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="News-to-Alpha Engine", layout="wide")

st.title("⚡ News-to-Alpha Arbitrage Engine")
st.markdown("""
**Hackathon MVP: Automated NLP Sentiment Trading**  
This engine scrapes live financial news, utilizes Deep Learning to score the sentiment (Bullish/Bearish) in milliseconds, and maps the resulting arbitrage signals directly against live market price action.
""")
st.divider()

# --- NLP MODEL INITIALIZATION ---
# We use cache_resource so the heavy AI model only loads into memory once
@st.cache_resource
def load_nlp_model():
    # Using a fast, pre-trained sentiment analysis model
    return pipeline("sentiment-analysis")

with st.spinner("Loading NLP Engine into memory..."):
    sentiment_analyzer = load_nlp_model()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Target Parameters")
ticker_symbol = st.sidebar.text_input("Target Ticker (e.g., AAPL, TSLA, NVDA)", value="TSLA")
confidence_threshold = st.sidebar.slider("AI Confidence Threshold", min_value=0.50, max_value=0.99, value=0.75, step=0.01)

if ticker_symbol:
    # --- DATA INGESTION ---
    stock = yf.Ticker(ticker_symbol)
    
    with st.spinner(f"Fetching market data and breaking news for {ticker_symbol}..."):
        # Fetch intraday market data (last 5 days, 15-minute intervals)
        hist_data = stock.history(period="5d", interval="15m")
        
        # Fetch latest news
        news_data = stock.news
        
    if hist_data.empty or not news_data:
        st.error("❌ Could not fetch data or news for this ticker. Try another symbol.")
    else:
        # --- NLP PROCESSING ---
        st.subheader("🧠 Live AI Sentiment Analysis")
        
        analyzed_news = []
        for article in news_data:
            title = article.get('title', '')
            # Convert UNIX timestamp to human-readable datetime
            pub_time = datetime.datetime.fromtimestamp(article.get('providerPublishTime', 0))
            
            # Run the headline through the NLP model
            if title:
                result = sentiment_analyzer(title)[0]
                sentiment = result['label']
                score = result['score']
                
                # Only keep signals that meet our confidence threshold
                if score >= confidence_threshold:
                    signal = "🟢 BULLISH" if sentiment == "POSITIVE" else "🔴 BEARISH"
                    analyzed_news.append({
                        "Datetime": pub_time,
                        "Headline": title,
                        "Sentiment": signal,
                        "AI Confidence": f"{score * 100:.1f}%"
                    })

        news_df = pd.DataFrame(analyzed_news)
        
        # Display the parsed news table
        if not news_df.empty:
            st.dataframe(news_df, use_container_width=True)
        else:
            st.info("No news met the current AI Confidence Threshold.")

        st.divider()

        # --- ARBITRAGE VISUALIZATION ---
        st.subheader("📊 Arbitrage Signal Mapping")
        
        fig = go.Figure()

        # 1. Candlestick Chart for Price Action
        fig.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name='Market Price'
        ))

        # 2. Overlay AI Signals
        # For a hackathon MVP, we map the news to the closest available timeframe
        if not news_df.empty:
            for _, row in news_df.iterrows():
                # Find the closest time in our historical data index
                closest_time = min(hist_data.index, key=lambda x: abs(x.tz_localize(None) - row['Datetime']))
                price_at_time = hist_data.loc[closest_time, 'Close']
                
                is_bullish = "BULLISH" in row['Sentiment']
                
                fig.add_trace(go.Scatter(
                    x=[closest_time],
                    y=[price_at_time],
                    mode='markers+text',
                    name='AI Signal',
                    marker=dict(
                        color='green' if is_bullish else 'red',
                        size=15,
                        symbol='triangle-up' if is_bullish else 'triangle-down',
                        line=dict(width=2, color='black')
                    ),
                    text="BUY" if is_bullish else "SELL",
                    textposition="top center" if is_bullish else "bottom center",
                    hoverinfo='text',
                    hovertext=f"{row['Headline']}<br>Confidence: {row['AI Confidence']}"
                ))

        fig.update_layout(
            title=f"{ticker_symbol} - AI Sentiment Overlay vs Price Action",
            yaxis_title="Stock Price",
            xaxis_title="Time",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # --- HACKATHON PITCH SUMMARY ---
        st.success("**Hackathon Pitch Value Proposition:** By executing trades in the millisecond window between the AI parsing the headline and human retail traders reacting to the news on their brokers, this engine captures pure, unpriced alpha.")