import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Financial Analyzer", page_icon="📈", layout="wide")

st.title("📈 AI-Powered Financial Analyzer")
st.markdown("Analyze stocks and cryptocurrencies with forecasting and technical indicators.")

def fetch_asset_data(ticker, period_years=2):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    asset = yf.Ticker(ticker)
    df = asset.history(start=start_date, end=end_date)
    if df.empty:
        return None
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df

def get_asset_info(ticker):
    asset = yf.Ticker(ticker)
    info = asset.info
    return {
        'ticker': ticker,
        'name': info.get('longName', info.get('shortName', ticker)),
        'type': 'crypto' if '-USD' in ticker else 'stock',
        'sector': info.get('sector', 'N/A'),
    }

def calculate_metrics(df):
    latest_price = df['Close'].iloc[-1]
    returns = {}
    for name, days in {'7d': 7, '30d': 30, '90d': 90}.items():
        if len(df) >= days:
            returns[name] = ((latest_price - df['Close'].iloc[-days]) / df['Close'].iloc[-days]) * 100
    daily_returns = df['Close'].pct_change()
    volatility_30d = daily_returns.tail(30).std() * (252 ** 0.5) * 100
    year_data = df.tail(252) if len(df) >= 252 else df
    return {
        'latest_price': latest_price,
        'returns': returns,
        'volatility_30d': volatility_30d,
        'high_52w': year_data['High'].max(),
        'low_52w': year_data['Low'].min(),
    }

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def add_technical_indicators(df):
    df = df.copy()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

def run_forecast(df, periods=30):
    prophet_df = df[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    current_price = df['Close'].iloc[-1]
    current_date = df['Date'].iloc[-1]
    future_forecast = forecast[forecast['ds'] > current_date].head(periods)
    end_price = future_forecast['yhat'].iloc[-1]
    price_change_pct = ((end_price - current_price) / current_price) * 100
    direction = "bullish" if price_change_pct > 2 else "bearish" if price_change_pct < -2 else "neutral"
    
    summary = {
        'predicted_price': end_price,
        'predicted_lower': future_forecast['yhat_lower'].iloc[-1],
        'predicted_upper': future_forecast['yhat_upper'].iloc[-1],
        'price_change_pct': price_change_pct,
        'direction': direction,
    }
    return forecast, summary

def plot_candlestick(df, forecast, ticker, metrics, summary, days=90):
    df_recent = df.tail(days).copy()
    
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f"{ticker} - Candlestick Chart", "Volume", "RSI", "MACD")
    )
    
    fig.add_trace(go.Candlestick(
        x=df_recent['Date'], open=df_recent['Open'], high=df_recent['High'],
        low=df_recent['Low'], close=df_recent['Close'], name='Price',
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['SMA_20'], mode='lines',
                             name='SMA 20', line=dict(color='#FFA726', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['SMA_50'], mode='lines',
                             name='SMA 50', line=dict(color='#42A5F5', width=1)), row=1, col=1)
    
    last_date = df['Date'].iloc[-1]
    future_fc = forecast[forecast['ds'] > last_date]
    fig.add_trace(go.Scatter(x=future_fc['ds'], y=future_fc['yhat'], mode='lines',
                             name='Forecast', line=dict(color='#AB47BC', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([future_fc['ds'], future_fc['ds'][::-1]]),
        y=pd.concat([future_fc['yhat_upper'], future_fc['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(171,71,188,0.2)', line=dict(color='rgba(0,0,0,0)'),
        name='Forecast Range'), row=1, col=1)
    
    colors = ['#26A69A' if df_recent['Close'].iloc[i] >= df_recent['Open'].iloc[i] else '#EF5350' 
              for i in range(len(df_recent))]
    fig.add_trace(go.Bar(x=df_recent['Date'], y=df_recent['Volume'], name='Volume',
                         marker_color=colors, showlegend=False), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['RSI'], mode='lines',
                             name='RSI', line=dict(color='#7E57C2', width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26A69A", line_width=1, row=3, col=1)
    
    fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['MACD'], mode='lines',
                             name='MACD', line=dict(color='#42A5F5', width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['MACD_Signal'], mode='lines',
                             name='Signal', line=dict(color='#FFA726', width=1.5)), row=4, col=1)
    hist_colors = ['#26A69A' if v >= 0 else '#EF5350' for v in df_recent['MACD_Hist']]
    fig.add_trace(go.Bar(x=df_recent['Date'], y=df_recent['MACD_Hist'], name='Histogram',
                         marker_color=hist_colors, showlegend=False), row=4, col=1)
    
    fig.update_layout(
        height=900, 
        title=f"{ticker} | ${metrics['latest_price']:.2f} | Forecast: ${summary['predicted_price']:.2f} ({summary['price_change_pct']:+.2f}%)",
        showlegend=True, hovermode='x unified', xaxis_rangeslider_visible=False
    )
    
    return fig

# Sidebar
st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input(
    "Enter Tickers (comma-separated)",
    value="AAPL, BTC-USD",
    help="Examples: AAPL, GOOGL, TSLA, BTC-USD, ETH-USD"
)

forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("**Popular Tickers:**")
st.sidebar.markdown("Stocks: AAPL, GOOGL, MSFT, TSLA, NVDA")
st.sidebar.markdown("Crypto: BTC-USD, ETH-USD, SOL-USD")

# Main
if st.sidebar.button("Run Analysis", type="primary"):
    
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        all_results = []
        progress = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            st.subheader(f"Analyzing {ticker}...")
            
            df = fetch_asset_data(ticker)
            
            if df is None:
                st.error(f"Could not fetch data for {ticker}")
                continue
            
            df = add_technical_indicators(df)
            info = get_asset_info(ticker)
            metrics = calculate_metrics(df)
            forecast, summary = run_forecast(df, forecast_days)
            
            all_results.append({
                'ticker': ticker,
                'info': info,
                'metrics': metrics,
                'summary': summary,
                'df': df
            })
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price", f"${metrics['latest_price']:.2f}")
            col2.metric("30D Return", f"{metrics['returns'].get('30d', 0):.2f}%")
            col3.metric("Forecast", f"${summary['predicted_price']:.2f}", f"{summary['price_change_pct']:+.2f}%")
            col4.metric("Signal", summary['direction'].upper())
            
            latest = df.iloc[-1]
            col5, col6, col7 = st.columns(3)
            col5.metric("RSI", f"{latest['RSI']:.1f}")
            col6.metric("MACD Signal", "Bullish" if latest['MACD'] > latest['MACD_Signal'] else "Bearish")
            col7.metric("Volatility", f"{metrics['volatility_30d']:.2f}%")
            
            fig = plot_candlestick(df, forecast, ticker, metrics, summary)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            progress.progress((i + 1) / len(tickers))
        
        if len(all_results) > 1:
            st.subheader("Comparison Table")
            comparison_data = []
            for r in all_results:
                latest = r['df'].iloc[-1]
                comparison_data.append({
                    'Ticker': r['ticker'],
                    'Price': f"${r['metrics']['latest_price']:.2f}",
                    '30D Return': f"{r['metrics']['returns'].get('30d', 0):.2f}%",
                    'Volatility': f"{r['metrics']['volatility_30d']:.2f}%",
                    'RSI': f"{latest['RSI']:.1f}",
                    'Forecast': f"${r['summary']['predicted_price']:.2f}",
                    'Change': f"{r['summary']['price_change_pct']:+.2f}%",
                    'Signal': r['summary']['direction'].upper()
                })
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        st.success("Analysis complete!")

st.sidebar.markdown("---")
st.sidebar.markdown("**Disclaimer:** For educational purposes only. Not financial advice.")