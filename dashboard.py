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

timeframes = ['1m', '5m', '15m', '1h', '1d']

# Initialize session state for auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 10

# Initialize session state for view mode
if 'show_only_signals' not in st.session_state:
    st.session_state.show_only_signals = False

# Initialize default times
default_start_time = time(9, 0)  # Market open at 9:00
default_end_time = time(17, 30)  # Market close at 17:30

# Initialize session state for time ranges
if 'start_time' not in st.session_state:
    st.session_state.start_time = default_start_time
if 'end_time' not in st.session_state:
    st.session_state.end_time = default_end_time

# Market timezone mapping
MARKET_TIMEZONES = {
    '^OMXS30': 'Europe/Stockholm',  # OMX Stockholm 30 (CET/CEST)
    '^GDAXI': 'Europe/Berlin',      # DAX Index (CET/CEST)
    '^GSPC': 'America/New_York',    # S&P 500 (ET)
    '^DJI': 'America/New_York'      # Dow Jones Industrial Average (ET)
}

def get_market_timezone(symbol):
    """Get the timezone for a given market symbol"""
    return MARKET_TIMEZONES.get(symbol, 'UTC')

def convert_to_market_timezone(dt, symbol):
    """Convert a datetime to the market's timezone"""
    market_tz = get_market_timezone(symbol)
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(pytz.timezone(market_tz))

def get_market_trading_hours(symbol):
    """Get the trading hours for a given market"""
    market_tz = get_market_timezone(symbol)
    tz = pytz.timezone(market_tz)
    
    # Default trading hours (can be customized per market)
    if symbol in ['^GSPC', '^DJI']:  # US Markets
        return {
            'start': time(9, 30),  # 9:30 AM ET
            'end': time(16, 0),    # 4:00 PM ET
            'timezone': market_tz
        }
    elif symbol == '^OMXS30':  # Swedish Market
        return {
            'start': time(9, 0),   # 9:00 AM CET
            'end': time(17, 30),   # 5:30 PM CET
            'timezone': market_tz
        }
    elif symbol == '^GDAXI':  # German Market
        return {
            'start': time(9, 0),   # 9:00 AM CET
            'end': time(17, 30),   # 5:30 PM CET
            'timezone': market_tz
        }
    else:
        return {
            'start': time(9, 0),
            'end': time(17, 0),
            'timezone': market_tz
        }

def style_signals_table(df):
    """Apply styling to the signals table"""
    # Create a copy of the dataframe to avoid modifying the original
    styled_df = df.copy()
    
    # Sort by timestamp in descending order (most recent first)
    styled_df = styled_df.sort_values('timestamp', ascending=False)
    
    # Format numeric columns
    numeric_cols = ['price', 'bb_upper', 'bb_middle', 'bb_lower', 'rsi']
    for col in numeric_cols:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) and x != '' else "")
    
    # Create a style dictionary
    styles = []
    
    # Enhanced styling for BUY signals
    buy_mask = styled_df['signal_strength'].str.contains('BUY', na=False)
    if buy_mask.any():
        styles.append({
            'selector': f'row:has(td:contains("BUY"))',
            'props': [
                ('background-color', '#e8f5e9'),  # Lighter green background
                ('color', '#1b5e20'),  # Darker green text
                ('font-weight', 'bold'),
                ('border-left', '5px solid #2e7d32'),  # Green border
                ('border-bottom', '1px solid #c8e6c9')  # Light green bottom border
            ]
        })
    
    # Enhanced styling for SELL signals
    sell_mask = styled_df['signal_strength'].str.contains('SELL', na=False)
    if sell_mask.any():
        styles.append({
            'selector': f'row:has(td:contains("SELL"))',
            'props': [
                ('background-color', '#ffebee'),  # Lighter red background
                ('color', '#b71c1c'),  # Darker red text
                ('font-weight', 'bold'),
                ('border-left', '5px solid #c62828'),  # Red border
                ('border-bottom', '1px solid #ffcdd2')  # Light red bottom border
            ]
        })
    
    # Add styling for BB positions
    styles.append({
        'selector': f'row:has(td:contains("Above Upper BB"))',
        'props': [
            ('background-color', '#ffebee'),  # Light red background
            ('color', '#c62828'),  # Dark red text
            ('font-weight', 'bold')
        ]
    })
    styles.append({
        'selector': f'row:has(td:contains("Below Lower BB"))',
        'props': [
            ('background-color', '#e8f5e9'),  # Light green background
            ('color', '#2e7d32'),  # Dark green text
            ('font-weight', 'bold')
        ]
    })
    styles.append({
        'selector': f'row:has(td:contains("Crossing Above Upper BB"))',
        'props': [
            ('background-color', '#fff3e0'),  # Light orange background
            ('color', '#e65100'),  # Dark orange text
            ('font-weight', 'bold'),
            ('font-style', 'italic')
        ]
    })
    styles.append({
        'selector': f'row:has(td:contains("Crossing Below Lower BB"))',
        'props': [
            ('background-color', '#e8f5e9'),  # Light green background
            ('color', '#2e7d32'),  # Dark green text
            ('font-weight', 'bold'),
            ('font-style', 'italic')
        ]
    })
    styles.append({
        'selector': f'row:has(td:contains("Touching"))',
        'props': [
            ('font-style', 'italic'),
            ('color', '#666666')  # Gray text
        ]
    })
    
    # Enhanced hover effects
    styles.append({
        'selector': 'tr:hover',
        'props': [
            ('background-color', '#fafafa'),
            ('transition', 'background-color 0.2s'),
            ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
        ]
    })
    
    # Add header styling
    styles.append({
        'selector': 'th',
        'props': [
            ('background-color', '#f5f5f5'),
            ('color', '#424242'),
            ('font-weight', 'bold'),
            ('text-align', 'left'),
            ('padding', '12px'),
            ('border-bottom', '2px solid #e0e0e0')
        ]
    })
    
    # Add cell padding and borders
    styles.append({
        'selector': 'td',
        'props': [
            ('padding', '8px 12px'),
            ('border-bottom', '1px solid #e0e0e0')
        ]
    })
    
    return styled_df.style.set_table_styles(styles)

def create_candlestick_chart(data, symbol, start_time=None, end_time=None, signals=None):
    """Create an interactive candlestick chart with Bollinger Bands and RSI"""
    # Get market timezone
    market_tz = get_market_timezone(symbol)
    
    # Ensure data is in market timezone
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert(market_tz)
    
    # Filter data based on time range if provided
    if start_time and end_time:
        # Convert time objects to datetime for comparison
        today = datetime.now().date()
        start_dt = datetime.combine(today, start_time)
        end_dt = datetime.combine(today, end_time)
        
        # Localize to market timezone
        start_dt = pytz.timezone(market_tz).localize(start_dt)
        end_dt = pytz.timezone(market_tz).localize(end_dt)
        
        # Filter data to only include rows within the time range
        mask = (data.index.time >= start_time) & (data.index.time <= end_time)
        data = data[mask]
        
        if data.empty:
            st.warning("No data available for the selected time range")
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
                            line=dict(color='rgba(255, 140, 0, 0.3)'),  # Orange with transparency
                            name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_middle'],
                            line=dict(color='rgba(128, 128, 128, 0.3)'),  # Gray with transparency
                            name='BB Middle'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'],
                            line=dict(color='rgba(30, 144, 255, 0.3)'),  # Blue with transparency
                            name='BB Lower'), row=1, col=1)

    # Find Bollinger Band crossovers
    upper_crosses = []
    lower_crosses = []
    
    for i in range(1, len(data)):
        prev_close = data['Close'].iloc[i-1]
        curr_close = data['Close'].iloc[i]
        prev_upper = data['bb_upper'].iloc[i-1]
        curr_upper = data['bb_upper'].iloc[i]
        prev_lower = data['bb_lower'].iloc[i-1]
        curr_lower = data['bb_lower'].iloc[i]
        
        # Upper band crossover (price crosses above upper band)
        if prev_close <= prev_upper and curr_close > curr_upper:
            upper_crosses.append({
                'time': data.index[i],
                'price': curr_close,
                'bb_value': curr_upper
            })
        
        # Lower band crossover (price crosses below lower band)
        if prev_close >= prev_lower and curr_close < curr_lower:
            lower_crosses.append({
                'time': data.index[i],
                'price': curr_close,
                'bb_value': curr_lower
            })

    # Add markers for upper band crossovers
    if upper_crosses:
        fig.add_trace(go.Scatter(
            x=[cross['time'] for cross in upper_crosses],
            y=[cross['price'] for cross in upper_crosses],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='#FF8C00',  # Dark Orange
                line=dict(width=1, color='#FF4500')  # Orange Red
            ),
            name='Upper BB Cross',
            hovertemplate='<b>Upper BB Cross</b><br>' +
                        'Time: %{x}<br>' +
                        'Price: %{y:.2f}<br>' +
                        'BB Value: %{customdata:.2f}<extra></extra>',
            customdata=[cross['bb_value'] for cross in upper_crosses]
        ), row=1, col=1)

    # Add markers for lower band crossovers
    if lower_crosses:
        fig.add_trace(go.Scatter(
            x=[cross['time'] for cross in lower_crosses],
            y=[cross['price'] for cross in lower_crosses],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='#1E90FF',  # Dodger Blue
                line=dict(width=1, color='#0000CD')  # Medium Blue
            ),
            name='Lower BB Cross',
            hovertemplate='<b>Lower BB Cross</b><br>' +
                        'Time: %{x}<br>' +
                        'Price: %{y:.2f}<br>' +
                        'BB Value: %{customdata:.2f}<extra></extra>',
            customdata=[cross['bb_value'] for cross in lower_crosses]
        ), row=1, col=1)

    # Add RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['rsi'],
                            line=dict(color='purple'),
                            name='RSI'), row=2, col=1)

    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Add buy and sell signals if available
    if signals:
        # Filter signals based on time range
        filtered_signals = []
        for signal in signals:
            if signal['timestamp'].tzinfo is None:
                signal['timestamp'] = pytz.UTC.localize(signal['timestamp'])
            signal['timestamp'] = signal['timestamp'].astimezone(pytz.timezone(market_tz))
            
            if start_time and end_time:
                signal_time = signal['timestamp'].time()
                if start_time <= signal_time <= end_time:
                    filtered_signals.append(signal)
            else:
                filtered_signals.append(signal)
        
        # Separate buy and sell signals
        buy_signals = [s for s in filtered_signals if s['action'] == 'BUY']
        sell_signals = [s for s in filtered_signals if s['action'] == 'SELL']
        
        # Add buy signals
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
        
        # Add sell signals
        if sell_signals:
            sell_times = [s['timestamp'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            sell_reasons = [', '.join(s['reason']) if isinstance(s['reason'], list) else s['reason'] for s in sell_signals]
            
            fig.add_trace(go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                name='Sell Signal',
                text=sell_reasons,
                hovertemplate='<b>Sell Signal</b><br>' +
                            'Time: %{x}<br>' +
                            'Price: %{y:.2f}<br>' +
                            'Reason: %{text}<extra></extra>'
            ), row=1, col=1)

    # Update layout with timezone information and range slider
    fig.update_layout(
        title=f'{symbol} Price Chart with Bollinger Bands and RSI ({market_tz})',
        yaxis_title='Price',
        yaxis2_title='RSI',
        xaxis_rangeslider_visible=True,
        height=800,
        margin=dict(l=20, r=20, t=70, b=20),
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
            tickformat='%Y-%m-%d %H:%M:%S',
            tickangle=45,
            rangeslider=dict(
                visible=True,
                thickness=0.1
            ),
            # Add range selector buttons
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=4, label="4h", step="hour", stepmode="backward"),
                    dict(count=8, label="8h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
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

def get_signals_for_market(symbol, timeframe, start_date, end_date, start_time=None, end_time=None):
    try:
        strategy = TradingStrategy(symbol, timeframe)
        
        # Get market trading hours
        trading_hours = get_market_trading_hours(symbol)
        market_tz = pytz.timezone(trading_hours['timezone'])
        
        # Use market trading hours as defaults if not specified
        if start_time is None:
            start_time = trading_hours['start']
        if end_time is None:
            end_time = trading_hours['end']
            
        # Create datetime objects in market timezone
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
        
        # Localize to market timezone
        start_datetime = market_tz.localize(start_datetime)
        end_datetime = market_tz.localize(end_datetime)
        
        # Fetch data for the selected date range
        data = strategy.fetch_data(start_date=start_date, end_date=end_date)
        if data is None or data.empty:
            st.error(f"No data available for {symbol} on {start_date}")
            return None, None
            
        # Ensure data is in market timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert(market_tz)
        
        # Calculate indicators before filtering by time range
        strategy.data = data
        strategy.calculate_indicators()
        
        # Filter data based on time range
        mask = (data.index >= start_datetime) & (data.index <= end_datetime)
        data = data[mask]
        
        if data.empty:
            st.warning(f"No data available for {symbol} in the selected time range")
            return None, None
        
        # Generate signals for the filtered data
        signals = strategy.generate_signals()
        
        # Convert signal timestamps to market timezone and filter by time range
        if signals:
            filtered_signals = []
            for signal in signals:
                if signal['timestamp'].tzinfo is None:
                    signal['timestamp'] = pytz.UTC.localize(signal['timestamp'])
                signal['timestamp'] = signal['timestamp'].astimezone(market_tz)
                
                # Only include signals within the time range
                if start_datetime <= signal['timestamp'] <= end_datetime:
                    filtered_signals.append(signal)
            signals = filtered_signals
        
        return signals, data
    except Exception as e:
        st.error(f"Error processing data for {symbol}: {str(e)}")
        return None, None

st.set_page_config(page_title="Trading Signals Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.title("📈 Trading Signals Dashboard")

# Add auto-refresh controls in a more compact layout
col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
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
with col4:
    st.session_state.show_only_signals = st.toggle("Show Only Signals", value=st.session_state.show_only_signals)

# Market selection
selected_market = st.selectbox(
    "Select Market",
    options=list(markets.keys()),
    format_func=lambda x: f"{x} ({markets[x]})"
)

# More compact date and time selection
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    # Get the current date
    current_date = datetime.now().date()
    
    # If it's weekend (5 = Saturday, 6 = Sunday), go back to Friday
    if current_date.weekday() >= 5:
        current_date = current_date - timedelta(days=current_date.weekday() - 4)
    
    selected_date = st.date_input("Date", current_date)
    
    # If user selects a weekend, automatically adjust to the previous Friday
    if selected_date.weekday() >= 5:
        selected_date = selected_date - timedelta(days=selected_date.weekday() - 4)
        st.info(f"Adjusted to previous trading day: {selected_date}")
with col2:
    selected_timeframe = st.selectbox("Timeframe", timeframes, index=0)
with col3:
    st.write("")  # Empty space for alignment
    st.write("")  # Empty space for alignment
    if st.button("Full Day"):
        st.session_state.start_time = time(9, 0)
        st.session_state.end_time = time(17, 30)
        st.rerun()

# Time range selection in a more compact layout
col1, col2 = st.columns(2)
with col1:
    start_time = st.time_input("Start", st.session_state.start_time, key="start_time_input")
    if start_time != st.session_state.start_time:
        st.session_state.start_time = start_time
        st.rerun()
with col2:
    end_time = st.time_input("End", st.session_state.end_time, key="end_time_input")
    if end_time != st.session_state.end_time:
        st.session_state.end_time = end_time
        st.rerun()

# Quick time range buttons in a single row
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Market Open"):
        st.session_state.start_time = time(9, 0)
        st.session_state.end_time = time(10, 30)
        st.rerun()
with col2:
    if st.button("Morning"):
        st.session_state.start_time = time(9, 0)
        st.session_state.end_time = time(12, 0)
        st.rerun()
with col3:
    if st.button("Afternoon"):
        st.session_state.start_time = time(13, 0)
        st.session_state.end_time = time(17, 0)
        st.rerun()

# Create a placeholder for the main content
main_content = st.empty()

# Main content update function
def update_content():
    with main_content.container():
        symbol = markets[selected_market]
        st.subheader(f"{selected_market} ({symbol})")
        
        # Get market trading hours
        trading_hours = get_market_trading_hours(symbol)
        
        # Use session state times
        current_start_time = st.session_state.start_time
        current_end_time = st.session_state.end_time
        
        signals, data = get_signals_for_market(
            symbol, 
            selected_timeframe, 
            selected_date, 
            selected_date + timedelta(days=1),
            current_start_time,
            current_end_time
        )
        
        if signals and data is not None and not data.empty:
            # Create the chart with current time range
            fig = create_candlestick_chart(data, symbol, start_time=current_start_time, end_time=current_end_time, signals=signals)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            
            # Filter signals based on time range
            signals_df = pd.DataFrame(signals)
            if not signals_df.empty:
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
                mask = (signals_df['timestamp'].dt.time >= current_start_time) & (signals_df['timestamp'].dt.time <= current_end_time)
                signals_df = signals_df[mask]
                
                # Filter for only signals if toggle is on
                if st.session_state.show_only_signals:
                    signals_df = signals_df[signals_df['action'].isin(['BUY', 'SELL'])]
                    if signals_df.empty:
                        st.info("No trading signals in the selected time range")
                        return
                
                # Add signal strength indicator
                signals_df['signal_strength'] = signals_df.apply(
                    lambda row: '🟢 BUY' if row['action'] == 'BUY' else '🔴 SELL' if row['action'] == 'SELL' else '⚪', 
                    axis=1
                )
                
                # Format timestamp
                signals_df['timestamp'] = signals_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Filter out rows with no signals and no reasons
                signals_df = signals_df[
                    (signals_df['action'].isin(['BUY', 'SELL'])) | 
                    (signals_df['reason'].apply(lambda x: len(x) > 0 if isinstance(x, list) else bool(x)))
                ]
                
                if signals_df.empty:
                    st.info("No significant signals or events in the selected time range")
                    return
                
                # Format RSI values
                signals_df['rsi'] = signals_df['rsi'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
                
                # Format reasons as bullet points
                signals_df['reason'] = signals_df['reason'].apply(
                    lambda x: "\n• " + "\n• ".join(x) if isinstance(x, list) and x else 
                            "• " + x if x else ""
                )
                
                # Reorder and select columns
                columns = ['signal_strength', 'timestamp', 'price', 'rsi', 'bb_position', 'reason']
                display_df = signals_df[columns].copy()
                
                # Rename columns for display
                display_df.columns = ['Signal', 'Time', 'Price', 'RSI', 'BB Position', 'Reasons']
                
                # Display the table
                st.markdown("### 🔔 Signals")
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    hide_index=True,
                    column_config={
                        "Signal": st.column_config.Column(width="small"),
                        "Time": st.column_config.Column(width="medium"),
                        "Price": st.column_config.Column(width="small"),
                        "RSI": st.column_config.Column(width="small"),
                        "BB Position": st.column_config.Column(width="medium"),
                        "Reasons": st.column_config.Column(width="large")
                    }
                )
        else:
            st.warning(f"No data available for {selected_market} on {selected_date}")
        st.markdown("---")

# Initial content update
update_content()

st.caption("Signals are for informational purposes only. Refresh to update.") 