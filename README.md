# Trading Signals Dashboard

A real-time trading signals dashboard built with Streamlit that provides technical analysis and trading signals for major market indices.

## Features

- Real-time market data from Yahoo Finance
- Technical indicators including Bollinger Bands and RSI
- Interactive candlestick charts with Plotly
- Trading signals with detailed analysis
- Support for multiple timeframes (1m, 5m, 15m, 1h, 1d)
- Market-specific trading hours and timezone handling
- Auto-refresh functionality

## Supported Markets

- OMX Stockholm 30 (^OMXS30)
- DAX Index (^GDAXI)
- S&P 500 (^GSPC)
- Dow Jones Industrial Average (^DJI)

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Local Development

Run the dashboard locally:
```bash
streamlit run dashboard.py
```

## Deployment

This dashboard is configured for deployment on Streamlit Community Cloud. To deploy:

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `dashboard.py` as your main file
5. Deploy!

## Configuration

The dashboard can be configured through the `.streamlit/config.toml` file. Current settings include:

- Custom theme colors
- Server security settings
- Browser settings

## License

MIT License 