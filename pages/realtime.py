import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trading import TradingStrategy
from datetime import datetime, timedelta, time
import pytz
import time as time_module

# Market symbols (Yahoo Finance)
markets = {
    'OMX': '^OMXS30',    # OMX Stockholm 30
    'DAX': '^GDAXI',     # DAX Index
    'SP500': '^GSPC',    # S&P 500
    'DJ': '^DJI'         # Dow Jones Industrial Average
}

timeframes = ['1m', '5m', '15m', '1h']

# Initialize session state for auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 60  # Refresh every minute
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = datetime.now().date()

def get_market_timezone(symbol):
    """Get the timezone for a given market symbol"""
    MARKET_TIMEZONES = {
        '^OMXS30': 'Europe/Stockholm',  # OMX Stockholm 30 (CET/CEST)
        '^GDAXI': 'Europe/Berlin',      # DAX Index (CET/CEST)
        '^GSPC': 'America/New_York',    # S&P 500 (ET)
        '^DJI': 'America/New_York'      # Dow Jones Industrial Average (ET)
    }
    return MARKET_TIMEZONES.get(symbol, 'UTC')

def get_latest_data(symbol, timeframe='1m', selected_date=None):
    """Get the latest data for a symbol"""
    strategy = TradingStrategy(symbol, timeframe)
    
    # Get current time in market timezone
    market_tz = pytz.timezone(get_market_timezone(symbol))
    now = datetime.now(market_tz)
    
    # Use selected date or current date
    if selected_date is None:
        selected_date = now.date()
    
    # If it's weekend, get Friday's data
    if selected_date.weekday() >= 5:
        selected_date = selected_date - timedelta(days=selected_date.weekday() - 4)
        st.info(f"Adjusted to previous trading day: {selected_date}")
    
    # Get data for the selected date
    start_date = selected_date
    end_date = start_date + timedelta(days=1)
    
    data = strategy.fetch_data(start_date=start_date, end_date=end_date)
    if data is not None and not data.empty:
        strategy.calculate_indicators()
        signals = strategy.generate_signals()
        return data, signals
    return None, None

def create_market_card(market_name, symbol, data, signals):
    """Create a card showing market data and signals"""
    if data is None or signals is None or data.empty:
        return f"""
        ### {market_name}
        **Status:** No data available
        """
    
    # Get current time in market timezone
    market_tz = pytz.timezone(get_market_timezone(symbol))
    now = datetime.now(market_tz)
    
    # Check if we're viewing current date or historical data
    is_current_date = st.session_state.selected_date == now.date()
    
    if is_current_date:
        # For current date, show last hour of data
        one_hour_ago = now - timedelta(hours=1)
        mask = data.index >= one_hour_ago
        recent_data = data[mask]
    else:
        # For historical data, show full day
        recent_data = data
    
    if recent_data.empty:
        return f"""
        ### {market_name}
        **Status:** No data available for selected date
        """
    
    try:
        # Create the card content with just the market name
        return f"""
        ### {market_name}
        """
    except Exception as e:
        return f"""
        ### {market_name}
        **Status:** Error processing data
        **Error:** {str(e)}
        """

def create_market_chart(market_name, symbol, data, signals):
    """Create a candlestick chart for a market"""
    if data is None or data.empty:
        return None
        
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.1,
                       row_heights=[0.70, 0.35],
                       subplot_titles=('Price', 'RSI'))

    # Add candlestick
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='OHLC'),
                  row=1, col=1)

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'],
                            line=dict(color='rgba(255, 140, 0, 0.3)'),
                            name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_middle'],
                            line=dict(color='rgba(128, 128, 128, 0.3)'),
                            name='BB Middle'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'],
                            line=dict(color='rgba(30, 144, 255, 0.3)'),
                            name='BB Lower'), row=1, col=1)

    # Add RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['rsi'],
                            line=dict(color='purple'),
                            name='RSI'), row=2, col=1)

    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Add buy signals if available
    if signals:
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        if buy_signals:
            buy_times = [s['timestamp'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            buy_reasons = [', '.join(s['reason']) if isinstance(s['reason'], list) else s['reason'] for s in buy_signals]
            
            fig.add_trace(go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                name='Buy Signal',
                text=buy_reasons,
                hovertemplate='<b>Buy Signal</b><br>' +
                            'Time: %{x}<br>' +
                            'Price: %{y:.2f}<br>' +
                            'Reason: %{text}<extra></extra>'
            ), row=1, col=1)

    # Update layout
    fig.update_layout(
        title=f'{market_name} ({symbol})',
        yaxis_title='Price',
        yaxis2_title='RSI',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            type='date',
            tickformat='%H:%M:%S',
            tickangle=45,
            rangeslider=dict(visible=False)
        ),
        yaxis2=dict(
            domain=[0, 0.3],
            anchor='x2'
        ),
        yaxis=dict(
            domain=[0.45, 1]
        )
    )

    return fig

def create_combined_signals_table(all_signals):
    """Create a combined table of all buy signals from all markets"""
    if not all_signals:
        return None
        
    # Combine all signals into one list
    combined_signals = []
    for market_name, signals in all_signals.items():
        if signals:
            for signal in signals:
                # Only include actual BUY or SELL signals
                if signal['action'] in ['BUY', 'SELL']:
                    combined_signals.append({
                        'Market': market_name,
                        'Signal': '🟢 BUY' if signal['action'] == 'BUY' else '🔴 SELL',
                        'Time': signal['timestamp'].strftime('%H:%M:%S'),
                        'Price': signal['price'],
                        'RSI': f"{signal['rsi']:.2f}" if pd.notnull(signal['rsi']) else "-",
                        'BB Position': signal['bb_position'],
                        'Reasons': ', '.join(signal['reason']) if isinstance(signal['reason'], list) else signal['reason']
                    })
    
    if not combined_signals:
        return None
        
    # Create DataFrame and sort by time
    df = pd.DataFrame(combined_signals)
    df = df.sort_values('Time', ascending=False)
    
    return df

def main():
    # Initialize session state for selected_date if not exists
    if 'selected_date' not in st.session_state:
        current_date = datetime.now().date()
        # If it's weekend (5 = Saturday, 6 = Sunday), go back to Friday
        if current_date.weekday() >= 5:
            current_date = current_date - timedelta(days=current_date.weekday() - 4)
        st.session_state.selected_date = current_date

    st.title("📊 Real-time Market Monitor")
    
    # Add date selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Get the current date
        current_date = datetime.now().date()
        
        # If it's weekend (5 = Saturday, 6 = Sunday), go back to Friday
        if current_date.weekday() >= 5:
            current_date = current_date - timedelta(days=current_date.weekday() - 4)
        
        selected_date = st.date_input("Date", st.session_state.selected_date)
        
        # If user selects a weekend, automatically adjust to the previous Friday
        if selected_date.weekday() >= 5:
            selected_date = selected_date - timedelta(days=selected_date.weekday() - 4)
            st.info(f"Adjusted to previous trading day: {selected_date}")
        
        # Update session state if date changed
        if selected_date != st.session_state.selected_date:
            st.session_state.selected_date = selected_date
            st.rerun()
    
    with col2:
        selected_timeframe = st.selectbox("Timeframe", timeframes, index=0)
    
    with col3:
        st.write("")  # Empty space for alignment
        st.write("")  # Empty space for alignment
        if st.button("Today"):
            st.session_state.selected_date = datetime.now().date()
            st.rerun()
    
    # Add auto-refresh controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        current_state = st.session_state.auto_refresh
        new_state = st.checkbox("Auto-refresh", value=current_state)
        if new_state != current_state:
            st.session_state.auto_refresh = new_state
            st.rerun()
    with col2:
        if st.session_state.auto_refresh:
            next_refresh = st.session_state.last_refresh + timedelta(seconds=st.session_state.refresh_interval)
            st.markdown(f"**Last:** {st.session_state.last_refresh.strftime('%H:%M:%S')} (Next: {next_refresh.strftime('%H:%M:%S')})")
        else:
            st.markdown(f"**Last:** {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    with col3:
        if st.button("🔄 Refresh"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # Create a placeholder for the main content
    main_content = st.empty()
    
    def update_content():
        with main_content.container():
            # First, collect all data and signals
            all_data = {}
            all_signals = {}
            all_cards = {}
            all_charts = {}
            
            for market_name, symbol in markets.items():
                data, signals = get_latest_data(symbol, selected_timeframe, st.session_state.selected_date)
                all_data[market_name] = data
                all_signals[market_name] = signals
                all_cards[market_name] = create_market_card(market_name, symbol, data, signals)
                all_charts[market_name] = create_market_chart(market_name, symbol, data, signals)
            
            # Show combined signals table
            st.markdown("### 🔔 Combined Buy Signals")
            signals_df = create_combined_signals_table(all_signals)
            if signals_df is not None:
                st.dataframe(
                    signals_df,
                    use_container_width=True,
                    height=200,
                    hide_index=True,
                    column_config={
                        "Market": st.column_config.Column(width="small"),
                        "Signal": st.column_config.Column(width="small"),
                        "Time": st.column_config.Column(width="small"),
                        "Price": st.column_config.Column(width="small"),
                        "RSI": st.column_config.Column(width="small"),
                        "BB Position": st.column_config.Column(width="medium"),
                        "Reasons": st.column_config.Column(width="large")
                    }
                )
            else:
                st.info("No buy signals available")
            
            st.markdown("---")
            
            # Show markets in a 2x2 grid
            st.markdown("### 📈 Market Charts")
            cols = st.columns(2)
            for i, (market_name, symbol) in enumerate(markets.items()):
                with cols[i % 2]:
                    st.markdown(all_cards[market_name])
                    if all_charts[market_name] is not None:
                        st.plotly_chart(all_charts[market_name], use_container_width=True)
                    st.markdown("---")
            
            is_current_date = st.session_state.selected_date == datetime.now().date()
            if is_current_date:
                st.caption("Data is refreshed every minute. Last hour of data is shown.")
            else:
                st.caption(f"Showing full day data for {st.session_state.selected_date}.")
    
    # Initial content update
    update_content()
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time_module.sleep(st.session_state.refresh_interval)
        st.session_state.last_refresh = datetime.now()
        st.rerun()

if __name__ == "__main__":
    main() 