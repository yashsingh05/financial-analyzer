# 📈 AI-Powered Financial Analyzer

A web application that analyzes stocks and cryptocurrencies using time series forecasting and technical indicators.

## 🚀 Live Demo

**[Click here to try the app](https://financial-analyzer-ffmmbbdpntf22b9dbq7m4v.streamlit.app)**

## 📸 Screenshot

![App Screenshot](https://via.placeholder.com/800x400?text=Financial+Analyzer+Screenshot)

## ✨ Features

- **Multi-Asset Support**: Analyze stocks (AAPL, GOOGL, TSLA) and cryptocurrencies (BTC-USD, ETH-USD)
- **Price Forecasting**: 7-60 day price predictions using Facebook Prophet
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Moving Averages (SMA 20, 50, 200)
- **Interactive Charts**: Professional candlestick charts with Plotly
- **Comparison Table**: Side-by-side comparison of multiple assets
- **Real-Time Data**: Fetches latest data from Yahoo Finance

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend language |
| Streamlit | Web application framework |
| Prophet | Time series forecasting |
| Plotly | Interactive visualizations |
| yfinance | Stock/crypto data API |
| Pandas | Data manipulation |

## 📊 How It Works

1. **Data Ingestion**: Fetches 2 years of historical price data from Yahoo Finance
2. **Technical Analysis**: Calculates RSI, MACD, and moving averages
3. **Forecasting**: Uses Facebook Prophet to predict future prices
4. **Visualization**: Displays candlestick charts with indicators and forecasts

## 🏃 Run Locally

```bash
# Clone the repository
git clone https://github.com/yashsingh05/financial-analyzer.git
cd financial-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
financial-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 📈 Sample Analysis

| Metric | Description |
|--------|-------------|
| Price | Current market price |
| 30D Return | Performance over last 30 days |
| RSI | Overbought (>70) or Oversold (<30) indicator |
| MACD Signal | Bullish or Bearish momentum |
| Forecast | Predicted price with confidence interval |
| Volatility | Annualized price volatility |

## 🔮 Future Enhancements

- [ ] Add AI-generated insights using Claude API
- [ ] Include news sentiment analysis
- [ ] Add portfolio tracking
- [ ] Export reports to PDF
- [ ] Add more technical indicators (Bollinger Bands, ATR)

## ⚠️ Disclaimer

This application is for **educational purposes only**. It is not financial advice. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

## 👤 Author

**Yash Singh**

- GitHub: [@yashsingh05](https://github.com/yashsingh05)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ If you found this project useful, please give it a star!
