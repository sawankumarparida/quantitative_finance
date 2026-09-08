import streamlit as st
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Monte Carlo DCF Valuation", layout="wide")

st.title("🎲 Monte Carlo DCF Valuation Engine")
st.markdown("""
**Management Consulting & M&A Scenario Modeler**  
Traditional models output a single, fragile valuation. This engine runs **10,000 parallel simulations** based on market volatility to output a probabilistic bell curve of a company's true intrinsic value.
""")
st.divider()

# --- SIDEBAR: ASSUMPTIONS & DRIVERS ---
st.sidebar.header("📊 Corporate Drivers")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, MSFT)", value="AAPL")

st.sidebar.subheader("Base Assumptions")
wacc = st.sidebar.slider("Discount Rate (WACC) %", min_value=5.0, max_value=15.0, value=9.0, step=0.5) / 100
tgr = st.sidebar.slider("Terminal Growth Rate %", min_value=1.0, max_value=5.0, value=2.5, step=0.1) / 100

st.sidebar.subheader("Risk & Volatility (Standard Deviation)")
st.sidebar.caption("How much variance do we expect in the 5-year forecast?")
rev_growth_mean = st.sidebar.slider("Expected Revenue Growth %", min_value=-5.0, max_value=25.0, value=8.0) / 100
rev_growth_std = st.sidebar.slider("Revenue Volatility (Std Dev) %", min_value=0.5, max_value=10.0, value=3.0) / 100

fcf_margin_mean = st.sidebar.slider("Expected FCF Margin %", min_value=5.0, max_value=40.0, value=22.0) / 100
fcf_margin_std = st.sidebar.slider("Margin Volatility (Std Dev) %", min_value=0.5, max_value=10.0, value=2.0) / 100

simulations = 10000

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_company_data(t):
    try:
        stock = yf.Ticker(t)
        info = stock.info
        current_price = info.get('currentPrice', info.get('previousClose', 100))
        shares_out = info.get('sharesOutstanding', 1000000000)
        # Fallback to estimated revenue if financials are missing
        total_rev = info.get('totalRevenue', 100000000000) 
        return current_price, shares_out, total_rev
    except Exception:
        return None, None, None

current_price, shares_out, base_revenue = fetch_company_data(ticker)

if current_price is None:
    st.error("❌ Could not fetch data for this ticker. Please check the symbol.")
else:
    # --- MONTE CARLO SIMULATION ENGINE ---
    # 1. Generate 10,000 random futures for Growth and Margin using Normal Distributions
    np.random.seed(42) # For reproducibility
    sim_rev_growth = np.random.normal(loc=rev_growth_mean, scale=rev_growth_std, size=simulations)
    sim_fcf_margin = np.random.normal(loc=fcf_margin_mean, scale=fcf_margin_std, size=simulations)

    # 2. Arrays to hold the final implied share prices
    implied_share_prices = np.zeros(simulations)

    # 3. The Vectorized DCF Math (Running 10k models simultaneously)
    # Project Revenue and FCF out 5 years
    rev_yr1 = base_revenue * (1 + sim_rev_growth)
    fcf_yr1 = rev_yr1 * sim_fcf_margin
    
    rev_yr2 = rev_yr1 * (1 + sim_rev_growth)
    fcf_yr2 = rev_yr2 * sim_fcf_margin
    
    rev_yr3 = rev_yr2 * (1 + sim_rev_growth)
    fcf_yr3 = rev_yr3 * sim_fcf_margin
    
    rev_yr4 = rev_yr3 * (1 + sim_rev_growth)
    fcf_yr4 = rev_yr4 * sim_fcf_margin
    
    rev_yr5 = rev_yr4 * (1 + sim_rev_growth)
    fcf_yr5 = rev_yr5 * sim_fcf_margin

    # Calculate Terminal Value (Gordon Growth Model)
    terminal_value = (fcf_yr5 * (1 + tgr)) / (wacc - tgr)

    # Discount everything back to Present Value (PV)
    pv_fcf1 = fcf_yr1 / (1 + wacc)**1
    pv_fcf2 = fcf_yr2 / (1 + wacc)**2
    pv_fcf3 = fcf_yr3 / (1 + wacc)**3
    pv_fcf4 = fcf_yr4 / (1 + wacc)**4
    pv_fcf5 = fcf_yr5 / (1 + wacc)**5
    pv_tv = terminal_value / (1 + wacc)**5

    # Calculate Enterprise Value (Simplified, excluding cash/debt for prototype)
    enterprise_value = pv_fcf1 + pv_fcf2 + pv_fcf3 + pv_fcf4 + pv_fcf5 + pv_tv
    
    # Calculate Per Share Value
    implied_share_prices = enterprise_value / shares_out

    # --- STATISTICAL ANALYSIS ---
    median_price = np.median(implied_share_prices)
    percentile_10 = np.percentile(implied_share_prices, 10)
    percentile_90 = np.percentile(implied_share_prices, 90)
    
    upside_downside = ((median_price / current_price) - 1) * 100

    # --- UI RENDER: EXECUTIVE METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Live Market Price", f"${current_price:,.2f}")
    col2.metric("Median Fair Value (AI)", f"${median_price:,.2f}", f"{upside_downside:.1f}% vs Market")
    col3.metric("Worst Case (10th Pct)", f"${percentile_10:,.2f}")
    col4.metric("Best Case (90th Pct)", f"${percentile_90:,.2f}")

    st.divider()

    # --- UI RENDER: PROBABILITY DISTRIBUTION CHART ---
    st.subheader(f"Valuation Probability Distribution ({simulations:,} Scenarios)")
    
    fig = px.histogram(
        x=implied_share_prices, 
        nbins=100, 
        color_discrete_sequence=['#1f77b4'],
        labels={'x': 'Implied Share Price ($)', 'y': 'Number of Scenarios'}
    )
    
    # Add vertical lines for context
    fig.add_vline(x=current_price, line_dash="dash", line_color="red", annotation_text="Current Market Price", annotation_position="top left")
    fig.add_vline(x=median_price, line_dash="dash", line_color="green", annotation_text="Median DCF Value", annotation_position="top right")
    
    fig.update_layout(showlegend=False, xaxis_title="Implied Share Price ($)", yaxis_title="Probability Density")
    st.plotly_chart(fig, use_container_width=True)

    # --- CONSULTING RECOMMENDATION ---
    st.subheader("📋 M&A Investment Recommendation")
    if median_price > current_price * 1.1:
        st.success(f"**STRONG BUY:** The statistical median values {ticker} at a premium to the market. The probability distribution suggests limited downside risk under current WACC and growth assumptions.")
    elif median_price < current_price * 0.9:
        st.error(f"**OVERVALUED:** {ticker} is currently trading at a premium to its intrinsic value. Re-evaluate M&A targets or adjust growth assumptions.")
    else:
        st.warning(f"**FAIRLY VALUED:** {ticker} is trading within the margin of error of its intrinsic value. No significant arbitrage opportunity detected.")